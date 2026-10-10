#!/usr/bin/env python3
"""
AstroSharp — Neural-network astronomical image sharpening for Siril
by Deep Sky Detail

Drop this file into your Siril scripts folder.
Put your weights/ folder in:
  <Siril config dir>/astrosharp/weights/
  (shown in the dialog when you first run the script)

Requires: Siril 1.4+, Python scripting enabled
"""

# ── Script metadata ───────────────────────────────────────────────────────────
#    name:        AstroSharp
#    description: Neural-network sharpening for astrophotography (Dual PSF,
#                 Hybrid, AstroClean, PSF, Second Beta, First Beta, Star Mask)
#    author:      Mark Lowry — Deep Sky Detail
#    contact:     https://www.youtube.com/@DeepSkyDetail
#    version:     1.1
#    requires:    1.4.0
#    licence:     MIT
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mark Lowry
#
# Model weights and training code:
#   https://github.com/deepskydetail/astrosharp-weights  (MIT licence)
#   https://github.com/deepskydetail/astrosharp          (MIT licence)

import sirilpy as s

# ── Install dependencies into Siril's venv (first run only) ──────────────────
s.ensure_installed("scipy", "scikit-image", "PyQt6",
                   version_constraints=[">=1.10", ">=0.21", ">=6.0"])

import sys, os, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import gaussian_filter, label as nd_label
from skimage.color import rgb2luv, luv2rgb
from typing import List

# ─────────────────────────────────────────────────────────────────────────────
# Neural-network forward pass
# ─────────────────────────────────────────────────────────────────────────────
class NeuralNet:
    """ReLU hidden layers, sigmoid output — matches the trained Keras weights."""
    def __init__(self, json_path: Path):
        with open(json_path) as f:
            data = json.load(f)
        self.linear_output: bool = data["linear_output"]
        self.layers: List[np.ndarray] = []
        for ld in data["weights"]:
            W = np.array(ld["values"], dtype=np.float64).reshape(ld["ncol"], ld["nrow"]).T
            self.layers.append(W)

    def predict(self, X: np.ndarray) -> np.ndarray:
        h = np.asarray(X, dtype=np.float64)
        n = len(self.layers)
        for i, W in enumerate(self.layers):
            z = h @ W[1:] + W[0]
            h = (1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
                 if i == n - 1 else np.maximum(0.0, z))
        return h.ravel().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────
_OFFSETS_9X9 = [
    (-1,-1),(-1, 0),(-1, 1),( 0,-1),( 0, 1),( 1,-1),( 1, 0),( 1, 1),( 0, 0),
    (-2,-2),(-2,-1),(-2, 0),(-2, 1),(-2, 2),(-1,-2),(-1, 2),( 0,-2),( 0, 2),
    ( 1,-2),( 1, 2),( 2,-2),( 2,-1),( 2, 0),( 2, 1),( 2, 2),
    (-4,-4),(-3,-4),(-2,-4),(-1,-4),( 0,-4),( 1,-4),( 2,-4),( 3,-4),( 4,-4),
    (-4,-3),(-3,-3),(-2,-3),(-1,-3),( 0,-3),( 1,-3),( 2,-3),( 3,-3),( 4,-3),
    (-4,-2),(-3,-2),( 3,-2),( 4,-2),(-4,-1),(-3,-1),( 3,-1),( 4,-1),
    (-4, 0),(-3, 0),( 3, 0),( 4, 0),( 4, 4),
    (-4, 1),(-3, 1),( 3, 1),( 4, 1),(-4, 2),(-3, 2),( 3, 2),( 4, 2),
    (-4, 3),(-3, 3),(-2, 3),(-1, 3),( 0, 3),( 1, 3),( 2, 3),( 3, 3),( 4, 3),
    (-4, 4),(-3, 4),(-2, 4),(-1, 4),( 0, 4),( 1, 4),( 2, 4),( 3, 4),
]
assert len(_OFFSETS_9X9) == 81
_SPATIAL_25 = _OFFSETS_9X9[:25]


def getmatrix9(img: np.ndarray) -> np.ndarray:
    nr, nc = img.shape
    r = np.arange(4, nr - 4, dtype=np.int32)
    c = np.arange(4, nc - 4, dtype=np.int32)
    R, C = np.meshgrid(r, c, indexing="ij")
    Rf, Cf = R.ravel(), C.ravel()
    out = np.empty((len(Rf), 81), dtype=np.float32)
    for k, (dr, dc) in enumerate(_OFFSETS_9X9):
        out[:, k] = img[Rf + dr, Cf + dc]
    return out


def getmatrix_fourier(img: np.ndarray) -> np.ndarray:
    nr, nc = img.shape
    r = np.arange(2, nr - 2, dtype=np.int32)
    c = np.arange(2, nc - 2, dtype=np.int32)
    R, C = np.meshgrid(r, c, indexing="ij")
    Rf, Cf = R.ravel(), C.ravel()
    ft = np.fft.fft2(img)
    out = np.empty((len(Rf), 27), dtype=np.float32)
    for k, (dr, dc) in enumerate(_SPATIAL_25):
        out[:, k] = img[Rf + dr, Cf + dc]
    out[:, 25] = ft.real[2:-2, 2:-2].astype(np.float32).ravel()
    out[:, 26] = ft.imag[2:-2, 2:-2].astype(np.float32).ravel()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Star mask
# ─────────────────────────────────────────────────────────────────────────────
def _hessian_det(blurred: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(blurred)
    gyy, _  = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    return gxx * gyy - gxy ** 2


def make_starmask(img, sensitivity, star_noise, feather, blend_gamma):
    scales = np.linspace(2, 20, 10)
    det_max = np.full_like(img, -1e30, dtype=np.float64)
    for s in scales:
        blurred = gaussian_filter(img.astype(np.float64), s)
        det_max = np.maximum(det_max, s ** 2 * _hessian_det(blurred))
    thresh = np.percentile(det_max, sensitivity)
    binary = (det_max >= thresh).astype(np.float32)
    labeled, _ = nd_label(binary)
    mask = (labeled > star_noise).astype(np.float32)
    mask = gaussian_filter(mask, max(feather, 0.01))
    return np.clip(mask ** blend_gamma, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction helpers
# ─────────────────────────────────────────────────────────────────────────────
def _scatter(preds, rows, cols):
    mat = np.zeros((int(rows.max()) + 1, int(cols.max()) + 1), dtype=np.float32)
    mat[rows.astype(np.int32), cols.astype(np.int32)] = preds
    return mat

def _extract_inner(filled):  return filled[8:-4, 8:-4]

def _place_result(inner, original):
    result = original.copy()
    r, c = inner.shape
    result[4:4 + r, 4:4 + c] = inner
    return result

def _chunk_seq(total, step):
    seq = list(range(1, total + 1, step))
    if seq[-1] != total:
        seq.append(total)
    return seq

def _chunk_bounds_9x9(rs0, rs1, cs0, cs1):
    r_start = 0 if rs0 == 1 else rs0 - 13
    c_start = 0 if cs0 == 1 else cs0 - 13
    r_out_s = rs0 + 3 if rs0 == 1 else rs0 - 9
    c_out_s = cs0 + 3 if cs0 == 1 else cs0 - 9
    return (r_start, rs1-1), (c_start, cs1-1), \
           np.arange(r_out_s, rs1-5, dtype=np.int32), \
           np.arange(c_out_s, cs1-5, dtype=np.int32)

def _chunk_bounds_fourier(rs0, rs1, cs0, cs1):
    r_start = 0 if rs0 == 1 else rs0 - 5
    c_start = 0 if cs0 == 1 else cs0 - 5
    r_out_s = rs0 + 1 if rs0 == 1 else rs0 - 3
    c_out_s = cs0 + 1 if cs0 == 1 else cs0 - 3
    return (r_start, rs1-1), (c_start, cs1-1), \
           np.arange(r_out_s, rs1-3, dtype=np.int32), \
           np.arange(c_out_s, cs1-3, dtype=np.int32)

def _scatter_results(results):
    return (np.concatenate([r[0] for r in results]),
            np.concatenate([r[1] for r in results]),
            np.concatenate([r[2] for r in results]))

_N_WORKERS = os.cpu_count() or 1


def _build_tasks_9x9(img_pad, chunk_size):
    nr, nc = img_pad.shape
    tasks = []
    for ci in range(1, len(_chunk_seq(nc, chunk_size))):
        colseq = _chunk_seq(nc, chunk_size)
        for ri in range(1, len(_chunk_seq(nr, chunk_size))):
            rowseq = _chunk_seq(nr, chunk_size)
            rs0, rs1 = rowseq[ri-1], rowseq[ri]
            cs0, cs1 = colseq[ci-1], colseq[ci]
            (r0,r1),(c0,c1), rr, cc = _chunk_bounds_9x9(rs0,rs1,cs0,cs1)
            if len(rr) == 0 or len(cc) == 0:
                continue
            tasks.append((img_pad[r0:r1, c0:c1].copy(), rr, cc))
    return tasks


def _build_tasks_fourier(img_pad, chunk_size):
    nr, nc = img_pad.shape
    tasks = []
    for ci in range(1, len(_chunk_seq(nc, chunk_size))):
        colseq = _chunk_seq(nc, chunk_size)
        for ri in range(1, len(_chunk_seq(nr, chunk_size))):
            rowseq = _chunk_seq(nr, chunk_size)
            rs0, rs1 = rowseq[ri-1], rowseq[ri]
            cs0, cs1 = colseq[ci-1], colseq[ci]
            (r0,r1),(c0,c1), rr, cc = _chunk_bounds_fourier(rs0,rs1,cs0,cs1)
            if len(rr) == 0 or len(cc) == 0:
                continue
            tasks.append((img_pad[r0:r1, c0:c1].copy(), rr, cc))
    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# Per-model pipelines
# ─────────────────────────────────────────────────────────────────────────────
def _run_single_9x9(img_pad, model, aggr, chunk_size):
    tasks = _build_tasks_9x9(img_pad, chunk_size)
    def _one(args):
        chunk, rr, cc = args
        feat = getmatrix9(chunk)
        pred = aggr * model.predict(feat) + (1.0 - aggr) * feat[:, 8]
        R, C = np.meshgrid(rr, cc, indexing="ij")
        return pred, R.ravel(), C.ravel()
    with ThreadPoolExecutor(max_workers=min(_N_WORKERS, len(tasks))) as pool:
        results = list(pool.map(_one, tasks))
    filled = _scatter(*_scatter_results(results))
    return _place_result(_extract_inner(filled), img_pad[4:-4, 4:-4])


def _run_dual_psf(img_pad, nn_dso, nn_stars, aggr, chunk_size,
                  sensitivity, star_noise, feather, blend_gamma):
    original = img_pad[4:-4, 4:-4].copy()
    tasks = _build_tasks_9x9(img_pad, chunk_size)
    def _one(args):
        chunk, rr, cc = args
        feat = getmatrix9(chunk)
        sm = make_starmask(chunk, sensitivity, star_noise, feather, blend_gamma)
        blended = nn_stars.predict(feat) * sm[4:-4,4:-4].ravel() + \
                  nn_dso.predict(feat) * (1.0 - sm[4:-4,4:-4].ravel())
        R, C = np.meshgrid(rr, cc, indexing="ij")
        return blended, R.ravel(), C.ravel()
    with ThreadPoolExecutor(max_workers=min(_N_WORKERS, len(tasks))) as pool:
        results = list(pool.map(_one, tasks))
    filled = _scatter(*_scatter_results(results))
    result = _place_result(_extract_inner(filled), original)
    return np.clip(result * aggr + original * (1.0 - aggr), 0.0, 1.0)


def _run_hybrid(img_pad, nn_psf, nn_clean, aggr, chunk_size):
    original = img_pad[4:-4, 4:-4].copy()
    tasks = _build_tasks_9x9(img_pad, chunk_size)
    def _one(args):
        chunk, rr, cc = args
        feat = getmatrix9(chunk)
        sharp = nn_psf.predict(feat)
        chunk2 = chunk.copy()
        rh, ch = chunk.shape
        chunk2[4:-4, 4:-4] = sharp.reshape(rh - 8, ch - 8)
        clean = nn_clean.predict(getmatrix9(chunk2))
        R, C = np.meshgrid(rr, cc, indexing="ij")
        return sharp, clean, R.ravel(), C.ravel()
    with ThreadPoolExecutor(max_workers=min(_N_WORKERS, len(tasks))) as pool:
        results = list(pool.map(_one, tasks))
    all_rows = np.concatenate([r[2] for r in results])
    all_cols = np.concatenate([r[3] for r in results])
    sharp_img = _place_result(_extract_inner(
        _scatter(np.concatenate([r[0] for r in results]), all_rows, all_cols)), original)
    clean_img = _place_result(_extract_inner(
        _scatter(np.concatenate([r[1] for r in results]), all_rows, all_cols)), original)
    return np.clip(aggr * clean_img + (1.0 - aggr) * sharp_img, 0.0, 1.0)


def _run_first_beta(img_pad, m1, m2, m3, m4, chunk_size):
    original = img_pad[2:-2, 2:-2].copy()
    tasks = _build_tasks_fourier(img_pad, chunk_size)
    def _one(args):
        chunk, rr, cc = args
        feat = getmatrix_fourier(chunk)
        pred = 0.70*m1.predict(feat) + 0.10*m2.predict(feat) + \
               0.10*m3.predict(feat) + 0.10*m4.predict(feat)
        R, C = np.meshgrid(rr, cc, indexing="ij")
        return pred, R.ravel(), C.ravel()
    with ThreadPoolExecutor(max_workers=min(_N_WORKERS, len(tasks))) as pool:
        results = list(pool.map(_one, tasks))
    filled = _scatter(*_scatter_results(results))
    inner = filled[4:-2, 4:-2]
    result = original.copy()
    r, c = inner.shape
    result[2:2+r, 2:2+c] = inner
    return np.clip(result, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level dispatcher
# ─────────────────────────────────────────────────────────────────────────────
def process_image(arr, model_name, color_mode, weights_dir,
                  chunk_size, aggr, psf_dso, psf_stars,
                  sensitivity, star_noise, feather, blend_gamma):
    weights_dir = Path(weights_dir)

    def load(name): return NeuralNet(weights_dir / name)
    def psf(v): return f"PSF_{v:g}.json"

    is_color = (arr.ndim == 3 and arr.shape[2] >= 3 and color_mode == "Color")

    if is_color:
        luv    = rgb2luv(arr[:, :, :3].astype(np.float64))
        tif1   = (luv[:, :, 0] / 100.0).astype(np.float32)
        luv_uv = luv[:, :, 1:].copy()
    else:
        tif1 = arr.astype(np.float32)
        if tif1.ndim == 3:
            tif1 = tif1[:, :, 0]

    if model_name == "Star Mask":
        result = make_starmask(tif1, sensitivity, star_noise, feather, blend_gamma)

    elif model_name in ("PSF Model (Pre-Beta)", "AstroClean", "Second Beta"):
        wfile = {"PSF Model (Pre-Beta)": psf(psf_dso),
                 "AstroClean": "AstroClean.json",
                 "Second Beta": "SecondBeta.json"}[model_name]
        result = _run_single_9x9(np.pad(tif1, 4, mode="constant"),
                                 load(wfile), aggr, chunk_size)

    elif model_name == "Dual PSF":
        result = _run_dual_psf(np.pad(tif1, 4, mode="constant"),
                               load(psf(psf_dso)), load(psf(psf_stars)),
                               aggr, chunk_size,
                               sensitivity, star_noise, feather, blend_gamma)

    elif model_name == "Hybrid Model":
        result = _run_hybrid(np.pad(tif1, 4, mode="constant"),
                             load(psf(psf_dso)), load("AstroClean.json"),
                             aggr, chunk_size)

    elif model_name == "First Beta":
        result = _run_first_beta(np.pad(tif1, 2, mode="constant"),
                                 load("FirstBeta_m1.json"), load("FirstBeta_m2.json"),
                                 load("FirstBeta_m3.json"), load("FirstBeta_m4.json"),
                                 chunk_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    result = np.clip(result, 0.0, 1.0)

    if is_color:
        luv_out = np.zeros((*result.shape, 3), dtype=np.float64)
        luv_out[:, :, 0] = result * 100.0
        luv_out[:, :, 1:] = luv_uv
        return np.clip(luv2rgb(luv_out), 0.0, 1.0).astype(np.float32)
    return result.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PARAMS = {
    "model_name":    "Dual PSF",
    "psf_dso":       3.0,
    "psf_stars":     3.0,
    "aggr":          1.0,
    "chunk_size":    325,
    "sensitivity":   98.0,
    "star_noise":    2.0,
    "feather":       5.0,
    "blend_gamma":   2.0,
}

def load_config(config_path: Path) -> dict:
    if config_path.exists():
        try:
            with open(config_path) as f:
                saved = json.load(f)
            params = DEFAULT_PARAMS.copy()
            params.update(saved)
            return params
        except Exception:
            pass
    return DEFAULT_PARAMS.copy()

def save_config(config_path: Path, params: dict):
    try:
        with open(config_path, "w") as f:
            json.dump(params, f, indent=2)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# GitHub weights download
# ─────────────────────────────────────────────────────────────────────────────
WEIGHTS_GITHUB_URL  = "https://github.com/deepskydetail/astrosharp-weights/raw/7e90c4ede09a6add69f1e9f8a5d2145a629067fa/weights.zip"
WEIGHTS_ZIP_SHA256  = "c74ba65b4095f78955b37cef0f3af9c4c258375a426b76bffb7715cd6f384e6b"
WEIGHTS_ZIP_SIZE_MB = 3.4

# ─────────────────────────────────────────────────────────────────────────────
# Model / visibility constants
# ─────────────────────────────────────────────────────────────────────────────
MODELS      = ["Dual PSF", "Hybrid Model", "PSF Model (Pre-Beta)",
               "AstroClean", "Second Beta", "First Beta", "Star Mask"]
PSF_MODELS  = {"PSF Model (Pre-Beta)", "Dual PSF", "Hybrid Model"}
DUAL_ONLY   = {"Dual PSF"}
AGGR_MODELS = {"PSF Model (Pre-Beta)", "Dual PSF", "Hybrid Model",
               "AstroClean", "Second Beta"}
MASK_MODELS = {"Dual PSF", "Star Mask"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
import shutil

def install_weights(src_dir: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in src_dir.glob("*.json"):
        shutil.copy2(f, dst_dir / f.name)
        count += 1
    return count

def available_psf(weights_dir: Path):
    found = []
    for v in np.arange(1, 8.25, 0.25):
        if (weights_dir / f"PSF_{v:g}.json").exists():
            found.append(round(float(v), 4))
    return found


# ─────────────────────────────────────────────────────────────────────────────
# PyQt6 stylesheet
# ─────────────────────────────────────────────────────────────────────────────
QSS = """
QDialog, QWidget {
    background-color: #0c0e19;
    color: #c8d8ee;
    font-family: Helvetica;
    font-size: 12px;
}
QGroupBox {
    color: #4299e1;
    font-weight: bold;
    font-size: 12px;
    border: 1px solid #1a3558;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 6px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}
QPushButton {
    background-color: #0c1929;
    color: #4299e1;
    border: 1px solid #1a3558;
    border-radius: 4px;
    padding: 5px 12px;
    font-weight: bold;
    font-size: 12px;
}
QPushButton:hover  { background-color: #122540; }
QPushButton:disabled { color: #2a4060; border-color: #0e1520; }
QPushButton#btn_process { font-size: 13px; padding: 7px 16px; }
QPushButton#btn_close   { font-size: 13px; padding: 7px 16px; }
QComboBox {
    background-color: #09090f;
    color: #c8d8ee;
    border: 1px solid #13192a;
    border-radius: 3px;
    padding: 3px 6px;
    font-size: 12px;
    min-height: 22px;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox::down-arrow { image: none; }
QComboBox QAbstractItemView {
    background-color: #09090f;
    color: #c8d8ee;
    selection-background-color: #13192a;
    selection-color: #4299e1;
    font-size: 12px;
}
QSlider::groove:horizontal {
    background: #13192a;
    height: 4px;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #4299e1;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background: #4299e1;
    border-radius: 2px;
}
QLabel  { color: #c8d8ee; }
QLabel#status_lbl { font-family: Courier; font-size: 13px; }
QLabel#path_lbl   { color: #c8d8ee; font-family: Courier; font-size: 10px; }
QLabel#ok_lbl     { color: #4ade80; font-size: 12px; }
QLabel#err_lbl    { color: #fc8181; font-size: 12px; }
QLabel#val_lbl    { color: #4299e1; font-size: 12px; min-width: 40px; }
QCheckBox { color: #c8d8ee; font-size: 12px; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #4299e1;
    background: #09090f;
    border-radius: 2px;
}
QCheckBox::indicator:checked { background-color: #4299e1; }
QProgressBar {
    background: #13192a;
    border: 1px solid #1a3558;
    border-radius: 3px;
    text-align: center;
    color: #c8d8ee;
    font-size: 11px;
    min-height: 16px;
}
QProgressBar::chunk { background: #4299e1; border-radius: 2px; }
"""


# ─────────────────────────────────────────────────────────────────────────────
# Background worker threads
# ─────────────────────────────────────────────────────────────────────────────
from PyQt6.QtCore import QThread, pyqtSignal


class ProcessWorker(QThread):
    finished = pyqtSignal()
    error    = pyqtSignal(str)

    def __init__(self, fn, params):
        super().__init__()
        self.fn     = fn
        self.params = params

    def run(self):
        try:
            self.fn(self.params)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class DownloadWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error    = pyqtSignal(str)

    def __init__(self, url: str, dest_dir: Path, expected_sha256: str = ""):
        super().__init__()
        self.url             = url
        self.dest_dir        = dest_dir
        self.expected_sha256 = expected_sha256

    def run(self):
        import urllib.request, zipfile, hashlib
        try:
            self.dest_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.dest_dir / "_weights_tmp.zip"

            def _hook(count, block, total):
                if total > 0:
                    self.progress.emit(min(int(count * block * 100 / total), 99))

            urllib.request.urlretrieve(self.url, str(tmp), _hook)

            # Verify SHA-256 before extracting
            if self.expected_sha256:
                sha = hashlib.sha256(tmp.read_bytes()).hexdigest()
                if sha != self.expected_sha256.lower():
                    tmp.unlink(missing_ok=True)
                    self.error.emit(
                        f"SHA-256 mismatch — download may be corrupt.\n"
                        f"Expected: {self.expected_sha256}\nGot:      {sha}")
                    return

            with zipfile.ZipFile(tmp, "r") as z:
                for member in z.namelist():
                    if member.endswith(".json"):
                        data = z.read(member)
                        out  = self.dest_dir / Path(member).name
                        out.write_bytes(data)

            tmp.unlink(missing_ok=True)
            self.progress.emit(100)
            self.finished.emit()
        except Exception as e:
            (self.dest_dir / "_weights_tmp.zip").unlink(missing_ok=True)
            self.error.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Main dialog
# ─────────────────────────────────────────────────────────────────────────────
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QComboBox, QSlider, QCheckBox,
    QProgressBar, QFileDialog, QMessageBox, QWidget, QSizePolicy,
    QScrollArea, QFrame,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui  import QFont


class AstroSharpDialog(QDialog):

    def __init__(self, config_path: Path, default_weights: Path,
                 do_process, get_psf=None):
        super().__init__()
        self.config_path     = config_path
        self.default_weights = default_weights
        self.do_process      = do_process
        self.get_psf         = get_psf
        self._worker         = None
        self._dl_worker      = None
        self._params         = load_config(config_path)

        self.setWindowTitle("AstroSharp — Deep Sky Detail")
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setMinimumWidth(500)
        self.setSizeGripEnabled(True)
        self.setStyleSheet(QSS)

        self._build_ui()
        self._refresh_weights_status()
        self._refresh_psf_dropdowns()
        self._on_model_changed()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(12, 12, 12, 12)

        # Weights ──────────────────────────────────────────────────────────────
        wg = QGroupBox("Weights")
        wl = QVBoxLayout(wg)

        self._weights_status = QLabel()
        self._weights_status.setWordWrap(True)
        wl.addWidget(self._weights_status)

        path_lbl = QLabel(str(self.default_weights))
        path_lbl.setObjectName("path_lbl")
        path_lbl.setWordWrap(True)
        wl.addWidget(path_lbl)

        btn_row = QHBoxLayout()
        btn_browse = QPushButton("Browse && Install")
        btn_browse.clicked.connect(self._browse_and_install)
        btn_dl = QPushButton("Download from GitHub")
        btn_dl.clicked.connect(self._download_from_github)
        btn_row.addWidget(btn_browse)
        btn_row.addWidget(btn_dl)
        wl.addLayout(btn_row)

        self._dl_bar = QProgressBar()
        self._dl_bar.setVisible(False)
        wl.addWidget(self._dl_bar)

        root.addWidget(wg)

        # Model ────────────────────────────────────────────────────────────────
        mg = QGroupBox("Model")
        ml = QVBoxLayout(mg)
        self._model_cb = QComboBox()
        self._model_cb.addItems(MODELS)
        idx = self._model_cb.findText(self._params.get("model_name", "Dual PSF"))
        if idx >= 0:
            self._model_cb.setCurrentIndex(idx)
        self._model_cb.currentTextChanged.connect(self._on_model_changed)
        self._model_cb.setFont(QFont("Helvetica", 12))
        ml.addWidget(self._model_cb)
        root.addWidget(mg)

        # PSF / Aggressiveness ─────────────────────────────────────────────────
        self._psf_group = QGroupBox("PSF / Aggressiveness")
        pg = QGridLayout(self._psf_group)
        pg.setColumnStretch(1, 1)

        self._row_dso,   self._psf_dso_cb,   _ = self._psf_row(
            pg, 0, "PSF DSO (\u03c3)", self._params.get("psf_dso", 3.0), "dso")
        self._row_stars, self._psf_stars_cb, _ = self._psf_row(
            pg, 1, "PSF Stars (\u03c3)", self._params.get("psf_stars", 3.0), "stars")
        self._row_aggr,  self._aggr_sl, self._aggr_lbl = self._slider_row(
            pg, 2, "Aggressiveness",
            1, 100, int(self._params.get("aggr", 1.0) * 100), 2, 0.01)

        root.addWidget(self._psf_group)

        # Processing ───────────────────────────────────────────────────────────
        proc_g = QGroupBox("Processing")
        prl = QGridLayout(proc_g)
        prl.setColumnStretch(1, 1)
        _, self._chunk_sl, self._chunk_lbl = self._slider_row(
            prl, 0, "Chunk Size (px)",
            50, 750, int(self._params.get("chunk_size", 325)), 1, 1)
        self._chunk_sl.valueChanged.connect(
            lambda v: self._chunk_lbl.setText(str(v)))
        self._chunk_lbl.setText(str(self._chunk_sl.value()))

        self._roi_cb = QCheckBox("Use current Siril selection as ROI (if set)")
        self._roi_cb.setChecked(self._params.get("use_roi", True))
        prl.addWidget(self._roi_cb, 1, 0, 1, 3)
        root.addWidget(proc_g)

        # Star Mask ────────────────────────────────────────────────────────────
        self._mask_group = QGroupBox("Star Mask Options")
        sml = QGridLayout(self._mask_group)
        sml.setColumnStretch(1, 1)

        _, self._sensi_sl,  self._sensi_lbl  = self._slider_row(
            sml, 0, "Sensitivity %",  1,  99, int(self._params.get("sensitivity", 98)), 1, 1)
        _, self._feather_sl, self._feather_lbl = self._slider_row(
            sml, 1, "Feathering (\u03c3)", 1, 200, int(self._params.get("feather", 5) * 10), 1, 0.1)
        _, self._gamma_sl,  self._gamma_lbl  = self._slider_row(
            sml, 2, "Blend Gamma",    1, 100, int(self._params.get("blend_gamma", 2.0) * 10), 1, 0.1)
        _, self._noise_sl,  self._noise_lbl  = self._slider_row(
            sml, 3, "Fix Mask (min)", 0,  20, int(self._params.get("star_noise", 2)), 1, 1)

        root.addWidget(self._mask_group)

        # Status ───────────────────────────────────────────────────────────────
        self._status_lbl = QLabel("")
        self._status_lbl.setObjectName("status_lbl")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setWordWrap(True)
        root.addWidget(self._status_lbl)

        # Buttons ──────────────────────────────────────────────────────────────
        btn_row2 = QHBoxLayout()
        self._btn_process = QPushButton("▶  Process")
        self._btn_process.setObjectName("btn_process")
        self._btn_process.clicked.connect(self._on_process)
        self._btn_close = QPushButton("Close")
        self._btn_close.setObjectName("btn_close")
        self._btn_close.clicked.connect(self.close)
        btn_row2.addWidget(self._btn_process)
        btn_row2.addWidget(self._btn_close)
        root.addLayout(btn_row2)

    def _psf_row(self, grid, row, label, default, key):
        container = QWidget()
        hl = QHBoxLayout(container)
        hl.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label)
        lbl.setFixedWidth(160)
        cb = QComboBox()
        cb.setFont(QFont("Helvetica", 12))
        hl.addWidget(lbl)
        hl.addWidget(cb)
        if self.get_psf is not None:
            btn = QPushButton("★ Auto")
            btn.setFixedWidth(80)
            btn.clicked.connect(lambda _=False, k=key: self._auto_psf(k))
            hl.addWidget(btn)
        grid.addWidget(container, row, 0, 1, 3)
        return container, cb, None

    def _slider_row(self, grid, row, label, mn, mx, default, col_span, scale):
        container = QWidget()
        hl = QHBoxLayout(container)
        hl.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label)
        lbl.setFixedWidth(160)
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setMinimum(mn)
        sl.setMaximum(mx)
        sl.setValue(default)
        val = QLabel(str(int(default)) if scale == 1 else str(round(default * scale, 2)))
        val.setObjectName("val_lbl")
        val.setFixedWidth(48)
        val.setAlignment(Qt.AlignmentFlag.AlignRight)
        sl.valueChanged.connect(
            lambda v, lv=val, sc=scale: lv.setText(
                str(int(v)) if sc == 1 else str(round(v * sc, 2))))
        hl.addWidget(lbl)
        hl.addWidget(sl)
        hl.addWidget(val)
        grid.addWidget(container, row, 0, 1, 3)
        return container, sl, val

    # ── Logic ─────────────────────────────────────────────────────────────────

    def _refresh_weights_status(self):
        jsons = list(self.default_weights.glob("*.json"))
        if jsons:
            self._weights_status.setText(f"✓  {len(jsons)} models installed")
            self._weights_status.setObjectName("ok_lbl")
        else:
            self._weights_status.setText("No weights installed — Browse & Install or Download from GitHub")
            self._weights_status.setObjectName("err_lbl")
        self._weights_status.setStyleSheet(
            "color: #4ade80;" if jsons else "color: #fc8181;")

    def _refresh_psf_dropdowns(self):
        vals = [str(v) for v in available_psf(self.default_weights)]
        for cb, key in ((self._psf_dso_cb, "psf_dso"),
                        (self._psf_stars_cb, "psf_stars")):
            cb.blockSignals(True)
            cb.clear()
            cb.addItems(vals)
            cur = str(self._params.get(key, 3.0))
            idx = cb.findText(cur)
            cb.setCurrentIndex(max(idx, 0))
            cb.blockSignals(False)

    def _on_model_changed(self):
        m = self._model_cb.currentText()

        def _show(widget, visible):
            widget.setVisible(visible)

        _show(self._psf_group,  m in PSF_MODELS or m in AGGR_MODELS)
        _show(self._mask_group, m in MASK_MODELS)

        # PSF rows within the group
        self._psf_dso_cb.parentWidget().setVisible(m in PSF_MODELS)
        self._psf_stars_cb.parentWidget().setVisible(m in DUAL_ONLY)
        self._aggr_sl.parentWidget().setVisible(m in AGGR_MODELS)

        self.adjustSize()

    def _auto_psf(self, key):
        if self.get_psf is None:
            return
        self._set_status("Detecting PSF…", "#f6ad55")
        sigma = self.get_psf()
        if sigma is None:
            self._set_status("No stars detected — run findstar first", "#fc8181")
            return
        fwhm = sigma * 2.355
        vals  = available_psf(self.default_weights)
        nearest = min(vals, key=lambda v: abs(v - sigma)) if vals else sigma
        cb = self._psf_dso_cb if key == "dso" else self._psf_stars_cb
        idx = cb.findText(str(nearest))
        if idx >= 0:
            cb.setCurrentIndex(idx)
        self._set_status(
            f"FWHM={fwhm:.2f}px  →  σ={sigma:.2f}  →  nearest model: {nearest}",
            "#4ade80")

    def _browse_and_install(self):
        src = QFileDialog.getExistingDirectory(
            self, "Select folder containing .json weight files")
        if not src:
            return
        src_path = Path(src)
        if not any(src_path.glob("*.json")):
            QMessageBox.critical(self, "AstroSharp",
                f"No .json weight files found in:\n{src}\n\n"
                "Use the AstroSharp export_weights.R script to generate them,\n"
                "or use Download from GitHub to fetch them automatically.")
            return
        n = install_weights(src_path, self.default_weights)
        self._refresh_weights_status()
        self._refresh_psf_dropdowns()
        self._set_status(f"Installed {n} weight files ✓", "#4ade80")

    def _download_from_github(self):
        reply = QMessageBox.question(
            self, "AstroSharp — Download Weights",
            f"This will download approximately {WEIGHTS_ZIP_SIZE_MB} MB of model "
            f"weight files from GitHub and install them to:\n\n{self.default_weights}\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._dl_bar.setVisible(True)
        self._dl_bar.setValue(0)
        self._btn_process.setEnabled(False)
        self._btn_close.setEnabled(False)
        self._set_status("Downloading weights…", "#f6ad55")

        self._dl_worker = DownloadWorker(WEIGHTS_GITHUB_URL, self.default_weights,
                                         WEIGHTS_ZIP_SHA256)
        self._dl_worker.progress.connect(self._dl_bar.setValue)
        self._dl_worker.finished.connect(self._on_download_done)
        self._dl_worker.error.connect(self._on_download_error)
        self._dl_worker.start()

    def _on_download_done(self):
        self._dl_bar.setVisible(False)
        self._btn_process.setEnabled(True)
        self._btn_close.setEnabled(True)
        self._refresh_weights_status()
        self._refresh_psf_dropdowns()
        self._set_status("Download complete ✓", "#4ade80")

    def _on_download_error(self, msg):
        self._dl_bar.setVisible(False)
        self._btn_process.setEnabled(True)
        self._btn_close.setEnabled(True)
        self._set_status(f"Download failed: {msg}", "#fc8181")

    def _collect_params(self) -> dict:
        return {
            "model_name":  self._model_cb.currentText(),
            "psf_dso":     float(self._psf_dso_cb.currentText() or 3.0),
            "psf_stars":   float(self._psf_stars_cb.currentText() or 3.0),
            "aggr":        round(self._aggr_sl.value() * 0.01, 2),
            "chunk_size":  int(self._chunk_sl.value()),
            "sensitivity": float(self._sensi_sl.value()),
            "star_noise":  float(self._noise_sl.value()),
            "feather":     round(self._feather_sl.value() * 0.1, 1),
            "blend_gamma": round(self._gamma_sl.value() * 0.1, 1),
            "use_roi":     self._roi_cb.isChecked(),
            "weights_dir": str(self.default_weights),
        }

    def _on_process(self):
        if not any(self.default_weights.glob("*.json")):
            QMessageBox.critical(self, "AstroSharp",
                "No weights installed.\n"
                "Use Browse & Install or Download from GitHub.")
            return

        params = self._collect_params()
        save_config(self.config_path, params)

        self._btn_process.setEnabled(False)
        self._btn_close.setEnabled(False)
        self._set_status("Processing…", "#f6ad55")

        self._worker = ProcessWorker(self.do_process, params)
        self._worker.finished.connect(self._on_process_done)
        self._worker.error.connect(self._on_process_error)
        self._worker.start()

    def _on_process_done(self):
        self._btn_process.setEnabled(True)
        self._btn_close.setEnabled(True)
        self._set_status("Done ✓", "#4ade80")

    def _on_process_error(self, msg):
        self._btn_process.setEnabled(True)
        self._btn_close.setEnabled(True)
        self._set_status(f"Error: {msg}", "#fc8181")

    def _set_status(self, text, colour):
        self._status_lbl.setText(text)
        self._status_lbl.setStyleSheet(
            f"color: {colour}; font-family: Courier; font-size: 13px;")

    def _is_busy(self):
        return ((self._worker is not None and self._worker.isRunning()) or
                (self._dl_worker is not None and self._dl_worker.isRunning()))

    def closeEvent(self, event):
        if self._is_busy():
            event.ignore()
        else:
            event.accept()

    def reject(self):
        if not self._is_busy():
            super().reject()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    siril = s.SirilInterface()
    try:
        siril.connect()
        siril.cmd("requires", "1.4.0")
    except s.SirilConnectionError as e:
        print(f"AstroSharp: cannot connect to Siril — {e}")
        return
    except Exception as e:
        print(f"AstroSharp: {e}")
        return

    if not siril.is_image_loaded():
        siril.error_messagebox("AstroSharp: No image loaded in Siril.\n"
                               "Please open an image first.")
        return

    config_dir      = Path(siril.get_siril_configdir()) / "astrosharp"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path     = config_dir / "settings.json"
    default_weights = config_dir / "weights"

    siril.log("AstroSharp by Deep Sky Detail")
    siril.log(f"  Config dir: {config_dir}")

    # ── do_process runs in a background thread ────────────────────────────────
    def do_process(params):
        siril.log(f"  Model:   {params['model_name']}")
        siril.log(f"  Weights: {params['weights_dir']}")

        with siril.image_lock():
            pixel_data = siril.get_image_pixeldata()

        siril.log(f"  Image shape: {pixel_data.shape}  dtype={pixel_data.dtype}")

        # Transpose (C, H, W) → (H, W, C) for colour images
        if pixel_data.ndim == 3 and pixel_data.shape[0] in (1, 3, 4):
            pixel_data = np.transpose(pixel_data, (1, 2, 0))

        channels   = pixel_data.shape[2] if pixel_data.ndim == 3 else 1
        color_mode = "Color" if channels >= 3 else "Black and White"

        scale_factor = 1.0
        if pixel_data.dtype == np.uint16:
            arr        = pixel_data.astype(np.float32) / 65535.0
            was_uint16 = True
        else:
            arr = pixel_data.astype(np.float32)
            if arr.max() > 1.5:
                scale_factor = float(arr.max())
                arr = arr / scale_factor
            was_uint16 = False

        sel = siril.get_siril_selection() if params.get("use_roi") else None
        if sel is not None:
            roi_x, roi_y, roi_w, roi_h = sel
            if roi_w > 0 and roi_h > 0:
                roi_y_np = arr.shape[0] - roi_y - roi_h
                siril.log(f"  ROI: x={roi_x} y={roi_y} w={roi_w} h={roi_h}")
                arr_process = arr[roi_y_np:roi_y_np+roi_h, roi_x:roi_x+roi_w]
            else:
                siril.log("  Selection cleared — processing full image")
                sel = None
                arr_process = arr
        else:
            siril.log("  No selection — processing full image")
            arr_process = arr

        result = process_image(
            arr_process,
            model_name   = params["model_name"],
            color_mode   = color_mode,
            weights_dir  = params["weights_dir"],
            chunk_size   = int(params["chunk_size"]),
            aggr         = float(params["aggr"]),
            psf_dso      = float(params["psf_dso"]),
            psf_stars    = float(params["psf_stars"]),
            sensitivity  = float(params["sensitivity"]),
            star_noise   = float(params["star_noise"]),
            feather      = float(params["feather"]),
            blend_gamma  = float(params["blend_gamma"]),
        )

        if sel is not None:
            out = (np.clip(arr, 0, 1) * 65535).astype(np.uint16) if was_uint16 \
                  else arr.astype(np.float32)
            region = (np.clip(result, 0, 1) * 65535).astype(np.uint16) if was_uint16 \
                     else result.astype(np.float32)
            out[roi_y_np:roi_y_np+roi_h, roi_x:roi_x+roi_w] = region
        else:
            out = (np.clip(result, 0, 1) * 65535).astype(np.uint16) if was_uint16 \
                  else result.astype(np.float32)

        if out.ndim == 3:
            out = np.transpose(out, (2, 0, 1))

        # Restore original scale for float images that were rescaled
        if not was_uint16 and scale_factor != 1.0:
            out = out * scale_factor

        siril.undo_save_state("AstroSharp")
        with siril.image_lock():
            siril.set_image_pixeldata(out)

        siril.log(f"  Done — model: {params['model_name']}")

    # ── PSF suggestion (runs synchronously before dialog opens) ──────────────
    def get_psf_suggestion():
        try:
            stars = siril.get_image_stars()
            if not stars:
                siril.cmd("findstar")
                stars = siril.get_image_stars()
            if not stars:
                return None
            fwhms = [(st.fwhmx + st.fwhmy) / 2.0
                     for st in stars if st.fwhmx > 0 and st.fwhmy > 0]
            return round(float(np.median(fwhms)) / 2.355, 2) if fwhms else None
        except Exception as e:
            siril.log(f"  PSF detection failed: {e}")
            return None

    # ── Launch Qt dialog ──────────────────────────────────────────────────────
    app = QApplication.instance() or QApplication([])
    dlg = AstroSharpDialog(config_path, default_weights,
                           do_process, get_psf=get_psf_suggestion)
    dlg.exec()


if __name__ == "__main__":
    main()
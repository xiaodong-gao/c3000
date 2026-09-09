#!/usr/bin/env python3
"""根据 DieMapSelect XML，从晶圆 PNG 中切割每一颗 die。"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

DEFAULT_DIR = Path(r"C:\Users\aspiring135\Desktop\datasets\10um-gray")
DETECT_TARGET = 8000
SKIP_DIRS = {"dies", "x64", "win32", "debug", "release", ".vs", ".git", "output"}


@dataclass
class DieMap:
    wafer_diameter: float
    edge_offset: float
    pitch_x: float
    pitch_y: float
    origin_offset_x: float
    origin_offset_y: float
    scribe_w: float
    scribe_h: float
    trim_cx: float
    trim_cy: float
    trim_size: float
    skipped: set[tuple[int, int]]


@dataclass
class Job:
    image: Path
    xml: Path
    out: Path
    name: str


def parse_die_map(xml_path: Path) -> DieMap:
    root = ET.parse(xml_path).getroot()
    layout = root.find("PatternedWaferLayout")
    if layout is None:
        raise ValueError(f"{xml_path} 中缺少 PatternedWaferLayout")
    trimming = root.find("Trimming")
    skipped = {
        (int(n.attrib["XIdx"]), int(n.attrib["YIdx"]))
        for n in layout.findall("SkippedDie")
    }
    return DieMap(
        wafer_diameter=float(layout.attrib["WaferDiameter"]),
        edge_offset=float(layout.attrib["EdgeOffset"]),
        pitch_x=float(layout.attrib["DiePitchX"]),
        pitch_y=float(layout.attrib["DiePitchY"]),
        origin_offset_x=float(layout.attrib["OriginalDieOffsetX"]),
        origin_offset_y=float(layout.attrib["OriginalDieOffsetY"]),
        scribe_w=float(layout.attrib["ScribeLineWidth"]),
        scribe_h=float(layout.attrib["ScribeLineHeight"]),
        trim_cx=float(trimming.attrib["CenterX"]) if trimming is not None else 0.0,
        trim_cy=float(trimming.attrib["CenterY"]) if trimming is not None else 0.0,
        trim_size=float(trimming.attrib["Size"]) if trimming is not None else 0.0,
        skipped=skipped,
    )


def detect_lut_width(gray: np.ndarray) -> int:
    col_mean = gray.mean(axis=0)
    width = 0
    while width < min(200, gray.shape[1]) and col_mean[width] > 100:
        width += 1
    return width


def detect_wafer_circle(gray: np.ndarray) -> tuple[float, float, float]:
    mask = gray > 65
    ys, xs = np.where(mask)
    if len(xs) < 1000:
        raise RuntimeError("未能从图像中找到晶圆区域")
    cx, cy = float(xs.mean()), float(ys.mean())
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    radius = float(np.percentile(dist, 99.5))
    keep = dist < radius * 1.02
    cx, cy = float(xs[keep].mean()), float(ys[keep].mean())
    dist = np.sqrt((xs[keep] - cx) ** 2 + (ys[keep] - cy) ** 2)
    radius = float(np.percentile(dist, 99.5))
    return cx, cy, radius


def _local_maxima(signal: np.ndarray, min_gap: int, min_value: float) -> list[int]:
    peaks: list[int] = []
    for i in range(2, len(signal) - 2):
        if signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1] and signal[i] > min_value:
            peaks.append(i)
    filtered: list[int] = []
    for p in peaks:
        nearby = [q for q in peaks if abs(q - p) <= min_gap]
        if all(signal[p] >= signal[q] for q in nearby) and (not filtered or p - filtered[-1] > min_gap):
            filtered.append(p)
    return filtered


def fit_regular_grid(peaks: list[int], period: float, span: int, tol: float) -> tuple[float, float, int]:
    peak_arr = np.array(peaks, dtype=np.float64)
    mid = peak_arr[(peak_arr > span * 0.15) & (peak_arr < span * 0.85)]
    if len(mid) < 3:
        mid = peak_arr
    seeds = mid[:: max(1, len(mid) // 24)]
    search = max(8.0, period * 0.12)
    best: tuple[int, float, float] | None = None
    for per in np.linspace(period - search, period + search, 49):
        if per <= 1:
            continue
        for p0 in seeds:
            n = np.round((peak_arr - p0) / per)
            hits = int(np.sum(np.abs(peak_arr - (p0 + n * per)) <= tol))
            if best is None or hits > best[0]:
                best = (hits, float(p0), float(per))
    assert best is not None
    _, p0, per = best
    inliers = peak_arr[np.abs(peak_arr - (p0 + np.round((peak_arr - p0) / per) * per)) <= tol]
    ns = np.round((inliers - p0) / per)
    a = np.column_stack([ns, np.ones_like(ns)])
    per, origin = np.linalg.lstsq(a, inliers, rcond=None)[0]
    hits = int(np.sum(np.abs(peak_arr - (origin + np.round((peak_arr - origin) / per) * per)) <= tol))
    origin = float(origin % per)
    if origin < 0:
        origin += per
    return origin, float(per), hits


def detect_vertical_streets(gray: np.ndarray, pitch_px: float) -> tuple[float, float]:
    h, w = gray.shape
    y0, y1 = int(h * 0.25), int(h * 0.75)
    col_p95 = np.percentile(gray[y0:y1, :], 95, axis=0)
    thr = max(float(np.percentile(col_p95, 82)), 120.0)
    peaks = _local_maxima(col_p95, min_gap=max(8, int(pitch_px * 0.25)), min_value=thr)
    if len(peaks) < 5:
        raise RuntimeError("垂直划片槽检测失败")
    origin, period, _hits = fit_regular_grid(peaks, pitch_px, w, tol=8)
    return origin, period


def detect_horizontal_bands(gray: np.ndarray, pitch_px: float, die_h_px: float) -> tuple[float, float, float]:
    h, w = gray.shape
    x0, x1 = int(w * 0.25), int(w * 0.75)
    row_p95 = np.percentile(gray[:, x0:x1], 95, axis=1)
    thr = max(float(np.percentile(row_p95, 78)), 120.0)
    peaks = _local_maxima(row_p95, min_gap=6, min_value=thr)
    if len(peaks) < 6:
        raise RuntimeError("水平划片槽检测失败")

    diffs = np.diff(peaks)
    die_gaps = [int(d) for d in diffs if abs(d - die_h_px) < max(12, die_h_px * 0.25)]
    die_h = float(np.median(die_gaps)) if die_gaps else die_h_px

    origin, period, _hits = fit_regular_grid(peaks, pitch_px, h, tol=10)
    best_origin = origin
    best_score = -1
    for shift in (0.0, period / 2):
        org = (origin + shift) % period
        score = 0
        for peak in peaks:
            n = round((peak - org) / period)
            dist = abs(peak - (org + n * period))
            dist2 = abs(peak - (org + n * period + die_h))
            if dist <= 10 or dist2 <= 10:
                score += 1
        if score > best_score:
            best_score = score
            best_origin = org

    tops: list[float] = []
    peak_arr = np.array(peaks, dtype=np.float64)
    n_lo = int(math.floor((0 - best_origin) / period))
    n_hi = int(math.ceil((h - best_origin) / period))
    for n in range(n_lo, n_hi + 1):
        top = best_origin + n * period
        bot = top + die_h
        if top < 0 or bot >= h:
            continue
        if np.min(np.abs(peak_arr - top)) <= 10 and np.min(np.abs(peak_arr - bot)) <= 12:
            tops.append(top)
    if tops:
        residuals = [t - (best_origin + round((t - best_origin) / period) * period) for t in tops]
        best_origin = (best_origin + float(np.mean(residuals))) % period
        die_h = float(np.median([
            min(peak_arr, key=lambda p: abs(p - (t + die_h))) - t
            for t in tops
        ]))
    return best_origin, period, abs(die_h)


def pick_step_mm(pitch: float, scribe: float, detected_mm: float) -> float:
    """DiePitch 有时含街道、有时是 die 本体。选更接近图像实测节距的那个。"""
    candidates = [pitch, pitch + scribe]
    return min(candidates, key=lambda s: abs(s - detected_mm))


def patterned_bbox(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """用局部方差找出真正有 die 图形的区域（半片空白晶圆也能用）。"""
    h, w = gray.shape
    tile = max(8, min(h, w) // 80)
    ny, nx = h // tile, w // tile
    if ny < 4 or nx < 4:
        return None
    stds = np.zeros((ny, nx), dtype=np.float64)
    means = np.zeros((ny, nx), dtype=np.float64)
    for iy in range(ny):
        for ix in range(nx):
            block = gray[iy * tile : (iy + 1) * tile, ix * tile : (ix + 1) * tile]
            stds[iy, ix] = float(block.std())
            means[iy, ix] = float(block.mean())
    pat = (stds > 8.0) & (means > 30.0) & (means < 220.0)
    col_frac = pat.mean(axis=0)
    row_frac = pat.mean(axis=1)
    cols = np.where(col_frac > 0.15)[0]
    rows = np.where(row_frac > 0.15)[0]
    if len(cols) < 3 or len(rows) < 3:
        return None
    x0 = int(cols[0] * tile)
    y0 = int(rows[0] * tile)
    x1 = int(min(w, (cols[-1] + 1) * tile))
    y1 = int(min(h, (rows[-1] + 1) * tile))
    if x1 - x0 < 80 or y1 - y0 < 80:
        return None
    return x0, y0, x1, y1


def autocorr_period(signal: np.ndarray, guess: float) -> float:
    """在 XML 猜测附近找自相关局部峰。窗口内的最大值常落在低频边缘，不能当节距。"""
    sig = signal.astype(np.float64)
    if len(sig) < 32 or guess <= 1:
        return guess
    win = max(5, int(round(guess)))
    kernel = np.ones(win) / win
    trend = np.convolve(sig, kernel, mode="same")
    s = sig - trend
    ac = np.correlate(s, s, mode="full")
    ac = ac[len(ac) // 2 :]
    lo = max(8, int(round(guess * 0.88)))
    hi = min(len(ac) - 2, int(round(guess * 1.12)))
    if hi <= lo + 2:
        return guess
    best_p: int | None = None
    best_v = -1e18
    for p in range(lo + 1, hi):
        if ac[p] >= ac[p - 1] and ac[p] >= ac[p + 1] and ac[p] > best_v:
            best_v = float(ac[p])
            best_p = p
    if best_p is None:
        return guess
    if abs(best_p - guess) / guess > 0.12:
        return guess
    return float(best_p)


def street_phase(signal: np.ndarray, period: float) -> int:
    """用局部极值拟合街道相位，避免半周期对准 die 中心。"""
    per = max(2, int(round(period)))
    sig = signal.astype(np.float64)
    gap = max(2, int(per * 0.6))
    mins: list[int] = []
    maxs: list[int] = []
    for i in range(2, len(sig) - 2):
        if sig[i] <= sig[i - 1] and sig[i] <= sig[i + 1]:
            mins.append(i)
        if sig[i] >= sig[i - 1] and sig[i] >= sig[i + 1]:
            maxs.append(i)

    def nms(peaks: list[int], values: np.ndarray, take_min: bool) -> list[int]:
        kept: list[int] = []
        for p in peaks:
            nearby = [q for q in peaks if abs(q - p) <= gap]
            if take_min:
                ok = all(values[p] <= values[q] for q in nearby)
            else:
                ok = all(values[p] >= values[q] for q in nearby)
            if ok and (not kept or p - kept[-1] > gap):
                kept.append(p)
        return kept

    dark_p = nms(mins, sig, True)
    bright_p = nms(maxs, sig, False)

    def residual(peaks: list[int]) -> tuple[float, float]:
        if len(peaks) < 3:
            return 1e18, 0.0
        phases = np.array([p % per for p in peaks], dtype=np.float64)
        # 圆周均值
        ang = phases / per * 2 * np.pi
        c = float(np.cos(ang).mean())
        s = float(np.sin(ang).mean())
        origin = (np.arctan2(s, c) / (2 * np.pi) * per) % per
        wrapped = (phases - origin + per / 2) % per - per / 2
        return float(np.mean(np.abs(wrapped))), float(origin)

    dark_err, dark_ph = residual(dark_p)
    bright_err, bright_ph = residual(bright_p)
    if dark_err <= bright_err:
        return int(round(dark_ph)) % per
    return int(round(bright_ph)) % per


def detect_die_grid(wafer: np.ndarray, pitch_x_guess: float, pitch_y_guess: float) -> tuple[float, float, float, float]:
    """在有图形的区域内检测 (street_x0, pitch_x, street_y0, pitch_y)。"""
    box = patterned_bbox(wafer)
    if box is None:
        region = wafer
        ox, oy = 0, 0
    else:
        x0, y0, x1, y1 = box
        region = wafer[y0:y1, x0:x1]
        ox, oy = x0, y0
    col_mean = region.mean(axis=0)
    row_mean = region.mean(axis=1)
    pitch_x = autocorr_period(col_mean, pitch_x_guess)
    pitch_y = autocorr_period(row_mean, pitch_y_guess)
    # 正方形 die：两轴统一到实测峰，避免一轴回退到 XML 中值猜测
    if abs(pitch_x_guess - pitch_y_guess) / max(pitch_x_guess, 1.0) < 0.08:
        x_fallback = abs(pitch_x - pitch_x_guess) < 1.0
        y_fallback = abs(pitch_y - pitch_y_guess) < 1.0
        if x_fallback and not y_fallback:
            pitch_x = pitch_y
        elif y_fallback and not x_fallback:
            pitch_y = pitch_x
        elif abs(pitch_x - pitch_y) / max((pitch_x + pitch_y) * 0.5, 1.0) > 0.02:
            pitch_x = pitch_y = 0.5 * (pitch_x + pitch_y)
    ph_x = street_phase(col_mean, pitch_x)
    ph_y = street_phase(row_mean, pitch_y)
    return float(ox + ph_x), float(pitch_x), float(oy + ph_y), float(pitch_y)


def mm_layout(xml: DieMap, detected_step_x_mm: float | None = None, detected_step_y_mm: float | None = None) -> tuple[float, float, float, float]:
    """返回 (step_x, step_y, die_w, die_h)，单位 mm。"""
    step_x = xml.pitch_x
    step_y = xml.pitch_y
    if detected_step_x_mm is not None:
        step_x = pick_step_mm(xml.pitch_x, xml.scribe_w, detected_step_x_mm)
    elif xml.scribe_w > xml.pitch_x * 0.35:
        step_x = xml.pitch_x + xml.scribe_w
    if detected_step_y_mm is not None:
        step_y = pick_step_mm(xml.pitch_y, xml.scribe_h, detected_step_y_mm)
    elif xml.scribe_h > xml.pitch_y * 0.35:
        step_y = xml.pitch_y + xml.scribe_h
    die_w = xml.pitch_x if abs(step_x - xml.pitch_x) > 1e-6 else max(xml.pitch_x - xml.scribe_w, xml.pitch_x * 0.5)
    die_h = xml.pitch_y if abs(step_y - xml.pitch_y) > 1e-6 else max(xml.pitch_y - xml.scribe_h, xml.pitch_y * 0.5)
    if abs(step_x - (xml.pitch_x + xml.scribe_w)) < 1e-6:
        die_w = xml.pitch_x
    if abs(step_y - (xml.pitch_y + xml.scribe_h)) < 1e-6:
        die_h = xml.pitch_y
    return step_x, step_y, die_w, die_h


@dataclass
class DieBox:
    x_idx: int
    y_idx: int
    left: int
    top: int
    right: int
    bottom: int
    skipped: bool


def infer_resolution_um(path: Path, default: float = 10.0) -> float:
    """从 10um-gray 这类路径名解析扫描分辨率。"""
    for text in (path.name, path.parent.name, str(path)):
        m = re.search(r"(\d+(?:\.\d+)?)\s*um", text, re.I)
        if m:
            return float(m.group(1))
    return default


def generate_dies(
    xml: DieMap,
    img_w: int,
    img_h: int,
    resolution_um: float,
    x_offset: int = 0,
) -> list[DieBox]:
    """按 C3000 DieMapLayout.CreateLayoutPoints 用 XML 几何生成 die 框。"""
    scale = 1000.0 / resolution_um
    center_x = img_w / 2.0 + xml.trim_cx * scale
    center_y = img_h / 2.0 - xml.trim_cy * scale

    die_w = xml.pitch_x
    die_h = xml.pitch_y
    step_x = die_w + xml.scribe_w
    step_y = die_h + xml.scribe_h
    offset_x = xml.origin_offset_x
    offset_y = -xml.origin_offset_y

    wafer_diameter = xml.wafer_diameter - xml.trim_size * 2.0
    wafer_radius = (wafer_diameter - xml.edge_offset * 2.0) / 2.0
    max_dist2 = wafer_radius * wafer_radius
    rect_w = int(die_w * scale)
    rect_h = int(die_h * scale)
    half_w = die_w / 2.0
    half_h = die_h / 2.0

    left_col = int(max(wafer_radius + offset_x, 0.0) / die_w) if die_w else 0
    top_row = int(max(wafer_radius + offset_y, 0.0) / die_h) if die_h else 0
    right_col = int(max(wafer_radius - offset_x, 0.0) / die_w) if die_w else 0
    bottom_row = int(max(wafer_radius - offset_y, 0.0) / die_h) if die_h else 0

    boxes: list[DieBox] = []
    for die_y in range(-top_row, bottom_row + 1):
        for die_x in range(-left_col, right_col + 1):
            x_mm = offset_x + die_x * step_x
            y_mm = offset_y + die_y * step_y
            inside = True
            for dx, dy in ((-half_w, -half_h), (half_w, -half_h), (-half_w, half_h), (half_w, half_h)):
                if (x_mm + dx) ** 2 + (y_mm + dy) ** 2 > max_dist2:
                    inside = False
                    break
            if not inside:
                continue
            x_px = round(x_mm * scale + center_x)
            y_px = round(y_mm * scale + center_y)
            left = int(x_px - rect_w / 2) + x_offset
            top = int(y_px - rect_h / 2)
            right = left + rect_w
            bottom = top + rect_h
            if right <= x_offset or bottom <= 0 or left >= img_w + x_offset or top >= img_h:
                continue
            left_c = max(left, x_offset)
            top_c = max(top, 0)
            right_c = min(right, x_offset + img_w)
            bottom_c = min(bottom, img_h)
            if right_c - left_c < 8 or bottom_c - top_c < 8:
                continue
            x_idx, y_idx = die_x, -die_y
            skipped = (die_x, die_y) in xml.skipped or (x_idx, y_idx) in xml.skipped
            boxes.append(
                DieBox(
                    x_idx=x_idx,
                    y_idx=y_idx,
                    left=left_c,
                    top=top_c,
                    right=right_c,
                    bottom=bottom_c,
                    skipped=skipped,
                )
            )
    boxes.sort(key=lambda b: (-b.y_idx, b.x_idx))
    return boxes


def scale_boxes(boxes: list[DieBox], scale: float) -> list[DieBox]:
    if scale == 1.0:
        return boxes
    out: list[DieBox] = []
    for b in boxes:
        out.append(
            DieBox(
                x_idx=b.x_idx,
                y_idx=b.y_idx,
                left=int(round(b.left * scale)),
                top=int(round(b.top * scale)),
                right=int(round(b.right * scale)),
                bottom=int(round(b.bottom * scale)),
                skipped=b.skipped,
            )
        )
    return out


def draw_overlay(full: Image.Image, boxes: list[DieBox], cx: float, cy: float, radius: float, x_offset: int) -> Image.Image:
    overlay = full.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    cx_full, cy_full = cx + x_offset, cy
    draw.ellipse(
        [cx_full - radius, cy_full - radius, cx_full + radius, cy_full + radius],
        outline=(0, 180, 255),
        width=2,
    )
    for box in boxes:
        color = (255, 80, 80) if box.skipped else (0, 255, 90)
        draw.rectangle([box.left, box.top, box.right - 1, box.bottom - 1], outline=color, width=1)
        if font is not None and (box.right - box.left) > 40:
            draw.text((box.left + 2, box.top + 2), f"{box.x_idx},{box.y_idx}", fill=color, font=font)
    return overlay


def make_detect_image(gray: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = gray.shape
    longest = max(h, w)
    if longest <= DETECT_TARGET:
        return gray, 1.0
    scale = longest / DETECT_TARGET
    new_w = max(1, int(round(w / scale)))
    new_h = max(1, int(round(h / scale)))
    small = np.array(Image.fromarray(gray).resize((new_w, new_h), Image.Resampling.BILINEAR))
    return small, scale


def xml_base_name(stem: str) -> str:
    lower = stem.lower()
    suffix = "diemapselect"
    if lower.endswith(suffix):
        stem = stem[: len(stem) - len(suffix)]
        return stem.rstrip("_- ")
    return stem


def is_cut_artifact_png(path: Path) -> bool:
    name = path.name.lower()
    if name == "die_overlay.png" or name.startswith("_dbg_"):
        return True
    return "die_x" in name and "_y" in name


def collect_files(folder: Path, recursive: bool) -> tuple[list[Path], list[Path]]:
    pngs: list[Path] = []
    xmls: list[Path] = []
    it = folder.rglob("*") if recursive else folder.iterdir()
    for p in it:
        if p.is_dir():
            if recursive and p.name.lower() in SKIP_DIRS:
                continue
            continue
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext == ".png" and not is_cut_artifact_png(p):
            pngs.append(p.resolve())
        elif ext == ".xml":
            xmls.append(p.resolve())
    return sorted(pngs), sorted(xmls)


def match_score(png_stem: str, xml_stem: str) -> int:
    p = png_stem.lower()
    x = xml_stem.lower()
    xb = xml_base_name(xml_stem).lower()
    if p == x or p == xb:
        return 300
    if x in {p + "diemapselect", p + "_diemapselect", p + "-diemapselect"}:
        return 280
    if xb and (xb in p or p in xb):
        return 100 + min(len(p), len(xb))
    if x in p or p in x:
        return 80
    return 0


def find_xml_for_png(png: Path, xmls: list[Path]) -> Path | None:
    stem = png.stem
    for cand in (
        png.with_suffix(".xml"),
        png.with_name(stem + "DieMapSelect.xml"),
        png.with_name(stem + "_DieMapSelect.xml"),
        png.with_name(stem + "-DieMapSelect.xml"),
    ):
        if cand.is_file():
            return cand.resolve()
    best = 0
    best_xml: Path | None = None
    for xml in xmls:
        score = match_score(stem, xml.stem)
        if score > best:
            best = score
            best_xml = xml
    return best_xml if best >= 100 else None


def collect_jobs(dir_path: Path, image: Path | None, xml: Path | None, out: Path | None, recursive: bool) -> list[Job]:
    if image is not None:
        name = image.stem
        xml_path = xml
        if xml_path is None:
            _pngs, xmls = collect_files(image.parent, False)
            xml_path = find_xml_for_png(image, xmls)
        if xml_path is None or not xml_path.is_file():
            raise RuntimeError(f"未找到与图像对应的 XML: {image.name}")
        out_dir = (image.parent / name) if out is None else (out / name)
        return [Job(image=image, xml=xml_path, out=out_dir, name=name)]

    folder = dir_path
    if not folder.is_dir():
        raise RuntimeError(f"不是有效文件夹: {folder}")
    pngs, xmls = collect_files(folder, recursive)
    if not pngs:
        raise RuntimeError(f"文件夹中没有可切割的 PNG: {folder}")
    if not xmls:
        raise RuntimeError(f"文件夹中没有 XML: {folder}")

    out_root = folder if out is None else out
    if len(pngs) == 1 and len(xmls) == 1 and find_xml_for_png(pngs[0], xmls) is None:
        return [Job(image=pngs[0], xml=xmls[0], out=out_root / pngs[0].stem, name=pngs[0].stem)]

    jobs: list[Job] = []
    for png in pngs:
        xml_path = find_xml_for_png(png, xmls)
        if xml_path is None:
            print(f"跳过（无对应 XML）: {png.name}")
            continue
        jobs.append(Job(image=png, xml=xml_path, out=out_root / png.stem, name=png.stem))
    if not jobs:
        raise RuntimeError("没有找到名字对应的 PNG/XML 配对")
    return jobs


def cut_dies(
    image_path: Path,
    xml_path: Path,
    out_dir: Path,
    keep_skipped: bool = False,
    resolution_um: float | None = None,
) -> None:
    xml = parse_die_map(xml_path)
    print("  读取图像...")
    image = Image.open(image_path)
    full = np.array(image.convert("L"))
    img_h, img_w = full.shape

    if resolution_um is None:
        resolution_um = infer_resolution_um(image_path)
    px_per_mm = 1000.0 / resolution_um

    detect, scale = make_detect_image(full)
    # LUT 必须在原图上测，避免缩小插值（PIL/OpenCV 不同）导致左右整体偏移
    lut = detect_lut_width(full)
    wafer_w = img_w - lut
    boxes = generate_dies(xml, wafer_w, img_h, resolution_um, x_offset=lut)

    wafer_radius_mm = (xml.wafer_diameter - xml.trim_size * 2.0 - xml.edge_offset * 2.0) / 2.0
    cx = lut + wafer_w / 2.0 + xml.trim_cx * px_per_mm
    cy = img_h / 2.0 - xml.trim_cy * px_per_mm
    radius = wafer_radius_mm * px_per_mm

    die_dir = out_dir / "dies"
    die_dir.mkdir(parents=True, exist_ok=True)
    for old in die_dir.glob("*.png"):
        try:
            old.unlink()
        except OSError:
            pass

    inv = 1.0 / scale
    boxes_s = scale_boxes(boxes, inv)
    overlay = draw_overlay(
        Image.fromarray(detect),
        boxes_s,
        cx * inv,
        cy * inv,
        radius * inv,
        0,
    )
    overlay_path = out_dir / "die_overlay.png"
    tmp_overlay = out_dir / "die_overlay.tmp.png"
    overlay.save(tmp_overlay)
    if overlay_path.exists():
        try:
            overlay_path.unlink()
        except OSError:
            overlay_path = out_dir / "die_overlay_new.png"
    tmp_overlay.replace(overlay_path)

    csv_path = out_dir / "die_map.csv"
    saved = 0
    skipped_n = 0
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x_idx", "y_idx", "left", "top", "right", "bottom", "width", "height", "skipped", "file"])
        for box in boxes:
            name = f"{image_path.stem}_die_x{box.x_idx}_y{box.y_idx}.png"
            if box.skipped and not keep_skipped:
                skipped_n += 1
                writer.writerow([box.x_idx, box.y_idx, box.left, box.top, box.right, box.bottom,
                                 box.right - box.left, box.bottom - box.top, 1, ""])
                continue
            left = max(box.left, 0)
            top = max(box.top, 0)
            right = min(box.right, img_w)
            bottom = min(box.bottom, img_h)
            if right - left < 8 or bottom - top < 8:
                continue
            crop = Image.fromarray(full[top:bottom, left:right])
            crop.save(die_dir / name)
            saved += 1
            skipped_n += int(box.skipped)
            writer.writerow([box.x_idx, box.y_idx, box.left, box.top, box.right, box.bottom,
                             box.right - box.left, box.bottom - box.top, int(box.skipped), name])

    print("图像:", image_path.name, f"{img_w}x{img_h}")
    print(f"分辨率: {resolution_um:g} um/px  LUT: {lut}px")
    print(f"布局圆心: ({cx:.1f}, {cy:.1f})  半径: {radius:.1f}px")
    print(f"die 尺寸: {xml.pitch_x}x{xml.pitch_y} mm  步进: {xml.pitch_x + xml.scribe_w}x{xml.pitch_y + xml.scribe_h} mm")
    print(f"原点偏移: ({xml.origin_offset_x}, {xml.origin_offset_y}) mm")
    print(f"有效 die: {saved}  跳过: {skipped_n}  输出: {die_dir}")
    print(f"预览: {overlay_path}")
    print(f"清单: {csv_path}")

    del image, full, detect, overlay
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description="根据 XML 晶圆布局切割 PNG 中的 die")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="输入文件夹（按文件名配对 PNG/XML）")
    parser.add_argument("--image", type=Path, default=None, help="单张晶圆 PNG")
    parser.add_argument("--xml", type=Path, default=None, help="对应 XML，省略则自动配对")
    parser.add_argument("--out", type=Path, default=None, help="输出根目录，默认写回输入文件夹")
    parser.add_argument("--recursive", "-r", action="store_true", help="递归扫描子目录")
    parser.add_argument("--keep-skipped", action="store_true", help="连 SkippedDie 一起导出")
    parser.add_argument("--resolution", type=float, default=None, help="扫描分辨率 um/px，默认从文件夹名解析")
    args = parser.parse_args()

    jobs = collect_jobs(args.dir, args.image, args.xml, args.out, args.recursive)
    print(f"共 {len(jobs)} 组 PNG/XML 待切割")
    failed = 0
    for i, job in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] {job.name}")
        print(f"  PNG: {job.image}")
        print(f"  XML: {job.xml}")
        print(f"  输出: {job.out}")
        try:
            cut_dies(
                job.image,
                job.xml,
                job.out,
                keep_skipped=args.keep_skipped,
                resolution_um=args.resolution,
            )
        except Exception as ex:
            print(f"  失败: {ex}")
            failed += 1
    if failed:
        raise SystemExit(f"完成，失败 {failed} / {len(jobs)}")
    print(f"\n全部完成: {len(jobs)} 组")


if __name__ == "__main__":
    main()

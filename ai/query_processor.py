"""Create robust views of a shop photo without destructive image edits."""

from __future__ import annotations

from pathlib import Path
import os
import threading

import numpy as np
from PIL import Image, ImageOps

try:
    import cv2
except ImportError:  # The application still supports basic search without it.
    cv2 = None


MAX_QUERY_SIZE = 1400
ISOLATED_CANVAS_SIZE = 1024
_rembg_session = None
_rembg_lock = threading.Lock()


def normalize_uploaded_image(source: str | Path, destination: str | Path) -> Path:
    """Apply phone EXIF orientation before converting HEIC/JPEG uploads."""
    destination = Path(destination)
    with Image.open(source) as opened:
        corrected = ImageOps.exif_transpose(opened).convert("RGB")
        corrected.thumbnail((2200, 2200), Image.Resampling.LANCZOS)
        corrected.save(destination, "JPEG", quality=95, optimize=True)
    return destination


def _ordered_corners(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    coordinate_sum = points.sum(axis=1)
    coordinate_difference = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(coordinate_sum)]
    ordered[2] = points[np.argmax(coordinate_sum)]
    ordered[1] = points[np.argmin(coordinate_difference)]
    ordered[3] = points[np.argmax(coordinate_difference)]
    return ordered


def _screen_view(image: Image.Image) -> Image.Image | None:
    """Detect and rectify a photographed monitor/display quadrilateral."""
    if cv2 is None:
        return None
    rgb = np.asarray(image)
    scale = min(1.0, 1000.0 / max(image.size))
    resized = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 135)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8), iterations=2
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = resized.shape[0] * resized.shape[1]
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.18 or area > image_area * 0.98:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        points = polygon.reshape(4, 2).astype(np.float32)
        rectangularity = area / max(float(cv2.contourArea(cv2.convexHull(points))), 1.0)
        candidates.append((area * rectangularity, points))
    if not candidates:
        return None

    points = _ordered_corners(max(candidates, key=lambda item: item[0])[1]) / scale
    top_left, top_right, bottom_right, bottom_left = points
    width = int(max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left)))
    height = int(max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left)))
    if width < 180 or height < 180:
        return None
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(points, destination)
    rectified = cv2.warpPerspective(rgb, matrix, (width, height))
    inset_x, inset_y = max(2, width // 50), max(2, height // 50)
    rectified = rectified[inset_y : height - inset_y, inset_x : width - inset_x]
    # A small downsample/upsample suppresses monitor pixel-grid moiré.
    reduced = cv2.resize(rectified, None, fx=0.72, fy=0.72, interpolation=cv2.INTER_AREA)
    rectified = cv2.resize(reduced, (rectified.shape[1], rectified.shape[0]), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(rectified, "RGB")


def _foreground_crop(image: Image.Image) -> Image.Image | None:
    """Estimate a product box from its difference to the outer image border."""
    preview = image.copy()
    preview.thumbnail((480, 480), Image.Resampling.LANCZOS)
    pixels = np.asarray(preview, dtype=np.float32)
    height, width = pixels.shape[:2]
    border_size = max(2, min(height, width) // 30)
    border = np.concatenate(
        (
            pixels[:border_size].reshape(-1, 3),
            pixels[-border_size:].reshape(-1, 3),
            pixels[:, :border_size].reshape(-1, 3),
            pixels[:, -border_size:].reshape(-1, 3),
        )
    )
    background = np.median(border, axis=0)
    distance = np.linalg.norm(pixels - background, axis=2)
    border_distance = np.linalg.norm(border - background, axis=1)
    threshold = max(30.0, float(np.percentile(border_distance, 97)) + 6.0)
    mask = distance > threshold

    dense_rows = np.where(mask.mean(axis=1) > 0.025)[0]
    dense_columns = np.where(mask.mean(axis=0) > 0.025)[0]
    if len(dense_rows) < 3 or len(dense_columns) < 3:
        return None
    left, right = int(dense_columns.min()), int(dense_columns.max()) + 1
    top, bottom = int(dense_rows.min()), int(dense_rows.max()) + 1
    box_area = (right - left) * (bottom - top)
    if box_area > width * height * 0.92:
        return None

    padding = max(8, int(max(right - left, bottom - top) * 0.14))
    left, top = max(0, left - padding), max(0, top - padding)
    right, bottom = min(width, right + padding), min(height, bottom + padding)
    scale_x, scale_y = image.width / width, image.height / height
    return image.crop(
        (
            round(left * scale_x),
            round(top * scale_y),
            round(right * scale_x),
            round(bottom * scale_y),
        )
    )


def _background_mask(image: Image.Image) -> Image.Image | None:
    """Return an AI foreground mask, while keeping background removal optional."""
    global _rembg_session
    model_directory = Path(__file__).resolve().parent.parent / "cache" / "models" / "u2net"
    os.environ.setdefault("U2NET_HOME", str(model_directory))
    try:
        from rembg import new_session, remove
    except ImportError:
        return None
    try:
        with _rembg_lock:
            if _rembg_session is None:
                _rembg_session = new_session("u2net")
            return remove(image, session=_rembg_session, only_mask=True).convert("L")
    except Exception:
        return None


def _isolated_product_views(
    image: Image.Image, mask: Image.Image
) -> tuple[Image.Image, Image.Image] | None:
    """Create a tight original crop and a centered white-background product view."""
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    alpha = np.asarray(mask, dtype=np.uint8)
    foreground = alpha >= 24
    rows, columns = np.where(foreground)
    if len(rows) < 64 or len(columns) < 64:
        return None
    left, right = int(columns.min()), int(columns.max()) + 1
    top, bottom = int(rows.min()), int(rows.max()) + 1
    box_area = (right - left) * (bottom - top)
    image_area = image.width * image.height
    if box_area > image_area * 0.90 or foreground.mean() < 0.002:
        return None

    padding = max(10, round(max(right - left, bottom - top) * 0.16))
    crop_box = (
        max(0, left - padding), max(0, top - padding),
        min(image.width, right + padding), min(image.height, bottom + padding),
    )
    focused = image.crop(crop_box)
    cutout = image.crop(crop_box).convert("RGBA")
    cutout.putalpha(mask.crop(crop_box))
    scale = min(820 / cutout.width, 820 / cutout.height)
    target_size = (
        max(1, round(cutout.width * scale)), max(1, round(cutout.height * scale))
    )
    cutout = cutout.resize(target_size, Image.Resampling.LANCZOS)
    white = Image.new("RGB", (ISOLATED_CANVAS_SIZE, ISOLATED_CANVAS_SIZE), "white")
    position = (
        (ISOLATED_CANVAS_SIZE - cutout.width) // 2,
        (ISOLATED_CANVAS_SIZE - cutout.height) // 2,
    )
    white.paste(cutout, position, cutout)
    return focused, white


def prepare_query_views(path: str | Path) -> list[Image.Image]:
    """Return original, tightly focused, and white-background product views."""
    with Image.open(path) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")
    original.thumbnail((MAX_QUERY_SIZE, MAX_QUERY_SIZE), Image.Resampling.LANCZOS)

    views = [original]
    screen = _screen_view(original)
    focus_source = screen if screen is not None else original
    mask = _background_mask(focus_source)
    isolated = _isolated_product_views(focus_source, mask) if mask is not None else None
    if isolated is not None:
        views.extend(isolated)
    else:
        cropped = _foreground_crop(focus_source)
        if cropped is not None and min(cropped.size) >= 48:
            views.append(cropped)
        elif screen is not None:
            views.append(screen)
    return views

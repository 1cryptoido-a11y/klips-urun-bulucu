"""Evaluate retrieval from simulated phone photographs of a monitor."""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ai.search_engine import ProductSearchEngine
from config import CATALOG_FILE, IMAGE_DIR


SAMPLE_SIZE = 30
SEED = 20260729


def screen_photo(source: Path, destination: Path, variant: int) -> None:
    product = cv2.imread(str(source))
    product = cv2.resize(product, (420, 300), interpolation=cv2.INTER_AREA)
    screen = np.full((570, 770, 3), 246, dtype=np.uint8)
    screen[125:425, 220:640] = product
    screen[38:82, 55:715] = (215, 215, 215)
    screen[112:500, 42:170] = (232, 232, 232)
    for row in range(450, 525, 18):
        screen[row : row + 6, 220:650] = (205, 205, 205)
    cv2.rectangle(screen, (0, 0), (769, 569), (18, 18, 18), 24)

    canvas = np.full((850, 1050, 3), (52, 47, 43), dtype=np.uint8)
    shifts = ((0, 0), (20, -15), (-18, 12))
    dx, dy = shifts[variant % len(shifts)]
    target = np.float32(
        [[115 + dx, 125 + dy], [925 + dx, 80 - dy], [970 - dx, 735 + dy], [70 - dx, 770 - dy]]
    )
    transform = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [769, 0], [769, 569], [0, 569]]), target
    )
    warped = cv2.warpPerspective(screen, transform, (canvas.shape[1], canvas.shape[0]))
    mask = cv2.warpPerspective(np.full(screen.shape[:2], 255, np.uint8), transform, (canvas.shape[1], canvas.shape[0]))
    canvas[mask > 0] = warped[mask > 0]
    for row in range(0, canvas.shape[0], 4):
        canvas[row : row + 1] = (canvas[row : row + 1].astype(np.float32) * 0.91).astype(np.uint8)
    glare = np.zeros_like(canvas)
    cv2.ellipse(glare, (760, 220), (170, 50), -18, 0, 360, (255, 255, 255), -1)
    canvas = cv2.addWeighted(canvas, 1.0, glare, 0.13, 0)
    cv2.imwrite(str(destination), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])


def rank(codes: list[str], expected: str) -> int:
    return codes.index(expected) + 1 if expected in codes else 999


def main() -> int:
    random.seed(SEED)
    catalog = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    products = [item for item in catalog if (IMAGE_DIR / f"{item['kod']}.jpg").is_file()]
    sample = random.sample(products, min(SAMPLE_SIZE, len(products)))
    engine = ProductSearchEngine()
    metrics = {"baseline_top1": 0, "baseline_top5": 0, "screen_top1": 0, "screen_top5": 0}
    with tempfile.TemporaryDirectory(prefix="klips-screen-test-") as directory:
        for number, product in enumerate(sample):
            query = Path(directory) / f"{product['kod']}.jpg"
            screen_photo(IMAGE_DIR / f"{product['kod']}.jpg", query, number)
            vectors = engine._query_vectors(query, "")
            scores, ids = engine.index.search(vectors[:1], 100)
            baseline_codes = [engine.codes[int(item_id)] for item_id in ids[0] if item_id >= 0]
            result_codes = [item["kod"] for item in engine.search(image_path=query, limit=8)]
            baseline_rank = rank(baseline_codes, product["kod"])
            screen_rank = rank(result_codes, product["kod"])
            metrics["baseline_top1"] += baseline_rank == 1
            metrics["baseline_top5"] += baseline_rank <= 5
            metrics["screen_top1"] += screen_rank == 1
            metrics["screen_top5"] += screen_rank <= 5
    print(f"SAMPLE={len(sample)}")
    for key, value in metrics.items():
        print(f"{key.upper()}={value}/{len(sample)}")
    return 0 if metrics["screen_top1"] >= metrics["baseline_top1"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

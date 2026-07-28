"""Compare single-view search with the robust multi-view search pipeline."""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from ai.search_engine import ProductSearchEngine
from config import CATALOG_FILE, IMAGE_DIR


SAMPLE_SIZE = 36
SEED = 20260728


def shop_like_photo(source: Path, destination: Path, variant: int) -> None:
    with Image.open(source) as opened:
        product = opened.convert("RGB")
    product.thumbnail((520, 520), Image.Resampling.LANCZOS)
    angle = (-9, -5, 5, 9)[variant % 4]
    product = product.rotate(angle, expand=True, fillcolor=(242, 239, 232))
    product = ImageEnhance.Brightness(product).enhance(0.88 + 0.08 * (variant % 3))
    rng = np.random.default_rng(SEED + variant)
    background = rng.normal(222, 15, (900, 1100, 3)).clip(0, 255).astype("uint8")
    canvas = Image.fromarray(background, "RGB")
    x = 80 + (variant * 47) % max(1, canvas.width - product.width - 120)
    y = 70 + (variant * 31) % max(1, canvas.height - product.height - 100)
    canvas.paste(product, (x, y))
    canvas.save(destination, "JPEG", quality=90)


def positions(results: list[dict], code: str) -> tuple[int, int]:
    codes = [item["kod"] for item in results]
    rank = codes.index(code) + 1 if code in codes else 999
    return int(rank == 1), int(rank <= 5)


def main() -> int:
    random.seed(SEED)
    catalog = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    eligible = [item for item in catalog if (IMAGE_DIR / f"{item['kod']}.jpg").is_file()]
    sample = random.sample(eligible, min(SAMPLE_SIZE, len(eligible)))
    engine = ProductSearchEngine()
    baseline_top1 = baseline_top5 = robust_top1 = robust_top5 = 0

    with tempfile.TemporaryDirectory(prefix="klips-benchmark-") as directory:
        for number, product in enumerate(sample):
            query = Path(directory) / f"{product['kod']}.jpg"
            shop_like_photo(IMAGE_DIR / f"{product['kod']}.jpg", query, number)
            vectors = engine._query_vectors(query, "")
            scores, ids = engine.index.search(vectors[:1], 100)
            baseline = [
                {**engine.products[engine.codes[int(item_id)]], "puan": float(score)}
                for item_id, score in zip(ids[0], scores[0])
                if item_id >= 0
            ][:8]
            robust = engine.search(image_path=query, limit=8)
            b1, b5 = positions(baseline, product["kod"])
            r1, r5 = positions(robust, product["kod"])
            baseline_top1 += b1
            baseline_top5 += b5
            robust_top1 += r1
            robust_top5 += r5

    total = len(sample)
    print(f"SAMPLE={total}")
    print(f"BASELINE_TOP1={baseline_top1}/{total}")
    print(f"BASELINE_TOP5={baseline_top5}/{total}")
    print(f"ROBUST_TOP1={robust_top1}/{total}")
    print(f"ROBUST_TOP5={robust_top5}/{total}")
    return 0 if robust_top1 >= baseline_top1 and robust_top5 >= baseline_top5 else 2


if __name__ == "__main__":
    raise SystemExit(main())

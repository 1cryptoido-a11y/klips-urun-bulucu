"""Caption all catalog images with Florence-2 using a resumable JSONL checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from PIL import Image, ImageFile, ImageOps
from transformers import AutoModelForCausalLM, AutoProcessor


ImageFile.LOAD_TRUNCATED_IMAGES = True
MODEL_ID = "microsoft/Florence-2-base-ft"
MODEL_REVISION = "f6c1a25888ffc1d945ee8a1a77ac833c7303d46e"
TASK = "<CAPTION>"


def image_map(directories: list[Path]) -> dict[str, Path]:
    mapped: dict[str, Path] = {}
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                mapped[path.stem] = path
    return mapped


def completed_codes(checkpoint: Path) -> set[str]:
    if not checkpoint.exists():
        return set()
    completed: set[str] = set()
    with checkpoint.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                completed.add(str(json.loads(line)["kod"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def batches(items: list[tuple[str, Path]], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    image.thumbnail((900, 900), Image.Resampling.LANCZOS)
    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args()

    catalog: list[dict] = json.loads(args.catalog.read_text(encoding="utf-8"))
    paths = image_map(args.image_dir)
    done = completed_codes(args.checkpoint)
    queue = [
        (str(product["kod"]), paths[str(product["kod"])])
        for product in catalog
        if str(product["kod"]) in paths and str(product["kod"]) not in done
    ]
    if args.max_items is not None:
        queue = queue[: args.max_items]
    print(f"CATALOG={len(catalog)} IMAGES={len(paths)} DONE={len(done)} QUEUE={len(queue)}", flush=True)
    if not queue:
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    processed = 0
    with args.checkpoint.open("a", encoding="utf-8") as output:
        for group in batches(queue, args.batch_size):
            valid: list[tuple[str, Path, Image.Image]] = []
            for code, path in group:
                try:
                    valid.append((code, path, load_image(path)))
                except Exception as exc:
                    output.write(json.dumps({"kod": code, "hata": str(exc)[:200]}, ensure_ascii=False) + "\n")
            if not valid:
                continue
            images = [item[2] for item in valid]
            inputs = processor(text=[TASK] * len(images), images=images, return_tensors="pt").to(
                device, dtype
            )
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, num_beams=args.beams
                )
            texts = processor.batch_decode(generated, skip_special_tokens=False)
            for (code, path, image), generated_text in zip(valid, texts):
                parsed = processor.post_process_generation(
                    generated_text, task=TASK, image_size=image.size
                )
                caption = str(parsed.get(TASK, "")).strip()
                caption = re.sub(r"<[^>]+>", " ", caption)
                caption = re.sub(r"\s+", " ", caption)
                output.write(
                    json.dumps(
                        {"kod": code, "caption_en": caption, "kaynak": str(path)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            output.flush()
            os.fsync(output.fileno())
            processed += len(valid)
            elapsed = time.perf_counter() - started
            rate = processed / elapsed if elapsed else 0.0
            remaining = (len(queue) - processed) / rate if rate else 0.0
            print(
                f"PROCESSED={processed}/{len(queue)} RATE={rate:.2f}/s ETA_MIN={remaining / 60:.1f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

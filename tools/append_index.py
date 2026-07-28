from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import open_clip
import torch
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


def batches(items: list[tuple[str, Path]], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-index", type=Path, required=True)
    parser.add_argument("--base-codes", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--output-codes", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    index = faiss.read_index(str(args.base_index))
    codes = [str(code) for code in json.loads(args.base_codes.read_text(encoding="utf-8"))]
    if index.ntotal != len(codes):
        raise ValueError(f"Index/code mismatch: {index.ntotal} != {len(codes)}")
    existing = set(codes)
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    queue = []
    for product in catalog:
        code = str(product["kod"])
        image_path = args.image_dir / f"{code}.jpg"
        if code not in existing and image_path.is_file():
            queue.append((code, image_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device).eval()
    appended = []
    with torch.inference_mode():
        for group in batches(queue, args.batch_size):
            tensors = []
            valid_codes = []
            for code, path in group:
                try:
                    with Image.open(path) as image:
                        tensors.append(preprocess(image.convert("RGB")))
                    valid_codes.append(code)
                except Exception as exc:
                    print(f"SKIP={code} ERROR={exc}")
            if not tensors:
                continue
            images = torch.stack(tensors).to(device)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            matrix = features.float().cpu().numpy().astype("float32")
            faiss.normalize_L2(matrix)
            index.add(matrix)
            appended.extend(valid_codes)
            print(f"APPENDED={len(appended)}/{len(queue)}", flush=True)

    output_codes = codes + appended
    if index.ntotal != len(output_codes):
        raise ValueError(f"Output mismatch: {index.ntotal} != {len(output_codes)}")
    faiss.write_index(index, str(args.output_index))
    args.output_codes.write_text(
        json.dumps(output_codes, ensure_ascii=False), encoding="utf-8"
    )
    print(f"INDEX={index.ntotal} NEW={len(appended)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

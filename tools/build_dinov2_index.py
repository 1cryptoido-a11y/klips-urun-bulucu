from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import faiss
import numpy as np
from PIL import Image, ImageFile
import timm
from timm.data import create_transform, resolve_model_data_config
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from tools.build_object_index import object_crop


ImageFile.LOAD_TRUNCATED_IMAGES = True
MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"


class DinoObjectCatalog(Dataset):
    def __init__(self, codes_file: Path, catalog_file: Path, image_dir: Path, transform) -> None:
        with codes_file.open("r", encoding="utf-8") as handle:
            self.codes = [str(code) for code in json.load(handle)]
        with catalog_file.open("r", encoding="utf-8") as handle:
            products = json.load(handle)
        self.categories = {
            str(product["kod"]): str(product.get("kategori", ""))
            for product in products
        }
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.codes)

    def __getitem__(self, index: int):
        code = self.codes[index]
        with Image.open(self.image_dir / f"{code}.jpg") as opened:
            image = object_crop(opened.convert("RGB"), self.categories.get(code, ""))
            return self.transform(image)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes-file", type=Path, required=True)
    parser.add_argument("--catalog-file", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    model = model.to(device).eval()
    transform = create_transform(
        **resolve_model_data_config(model), is_training=False
    )
    dataset = DinoObjectCatalog(
        args.codes_file, args.catalog_file, args.image_dir, transform
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
    )
    vectors: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_number, images in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                features = model(images)
            features = features / features.norm(dim=-1, keepdim=True)
            vectors.append(features.float().cpu().numpy())
            if batch_number % 20 == 0:
                print(
                    f"Processed {min(batch_number * args.batch_size, len(dataset)):,}/{len(dataset):,}",
                    flush=True,
                )

    matrix = np.concatenate(vectors).astype("float32")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    faiss.write_index(index, str(temporary))
    temporary.replace(args.output)
    print(f"DINOv2 FAISS ready: {index.ntotal:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import faiss
import numpy as np
import open_clip
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

from config import CATALOG_FILE, CODES_FILE, IMAGE_DIR, INDEX_FILE, MODEL_NAME, PRETRAINED


ImageFile.LOAD_TRUNCATED_IMAGES = True


class CatalogImages(Dataset):
    def __init__(self, preprocess) -> None:
        with CATALOG_FILE.open("r", encoding="utf-8") as handle:
            products = json.load(handle)
        self.items = [
            (str(product["kod"]), IMAGE_DIR / f"{product['kod']}.jpg")
            for product in products
            if (IMAGE_DIR / f"{product['kod']}.jpg").exists()
        ]
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        code, path = self.items[index]
        with Image.open(path) as image:
            return self.preprocess(image.convert("RGB")), code


def save_json_atomic(path: Path, value) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as handle:
        json.dump(value, handle, ensure_ascii=False)
        temp = Path(handle.name)
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device).eval()
    dataset = CatalogImages(preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    vectors: list[np.ndarray] = []
    codes: list[str] = []

    with torch.inference_mode():
        for batch_number, (images, batch_codes) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            vectors.append(features.float().cpu().numpy())
            codes.extend(batch_codes)
            if batch_number % 10 == 0:
                print(f"İşlenen: {len(codes):,}/{len(dataset):,}")

    matrix = np.concatenate(vectors).astype("float32")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    temp_index = INDEX_FILE.with_suffix(".index.tmp")
    faiss.write_index(index, str(temp_index))
    temp_index.replace(INDEX_FILE)
    save_json_atomic(CODES_FILE, codes)
    print(f"FAISS hazır: {index.ntotal:,} ürün")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

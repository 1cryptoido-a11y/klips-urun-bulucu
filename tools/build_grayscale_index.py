from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import faiss
import numpy as np
import open_clip
from PIL import Image, ImageFile, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import MODEL_NAME, PRETRAINED


ImageFile.LOAD_TRUNCATED_IMAGES = True


class GrayscaleCatalog(Dataset):
    def __init__(self, codes_file: Path, image_dir: Path, preprocess) -> None:
        with codes_file.open("r", encoding="utf-8") as handle:
            self.codes = [str(code) for code in json.load(handle)]
        self.image_dir = image_dir
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.codes)

    def __getitem__(self, index: int):
        code = self.codes[index]
        with Image.open(self.image_dir / f"{code}.jpg") as opened:
            image = ImageOps.autocontrast(ImageOps.grayscale(opened)).convert("RGB")
            return self.preprocess(image)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes-file", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model = model.to(device).eval()
    dataset = GrayscaleCatalog(args.codes_file, args.image_dir, preprocess)
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
                features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            vectors.append(features.float().cpu().numpy())
            if batch_number % 20 == 0:
                print(f"Processed {min(batch_number * args.batch_size, len(dataset)):,}/{len(dataset):,}", flush=True)

    matrix = np.concatenate(vectors).astype("float32")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    faiss.write_index(index, str(temporary))
    temporary.replace(args.output)
    print(f"Grayscale FAISS ready: {index.ntotal:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

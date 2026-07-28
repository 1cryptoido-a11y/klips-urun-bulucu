"""Generate conservative Turkish visual descriptions from CLIP embeddings."""

from __future__ import annotations

import json
import shutil

import faiss
import numpy as np
import open_clip
import torch

from config import CATALOG_FILE, CODES_FILE, INDEX_FILE, MODEL_NAME, PRETRAINED
from tools.import_catalog import write_json_atomic


ATTRIBUTE_GROUPS = {
    "renk": [
        ("altın tonlu", "a gold-tone jewelry product"),
        ("gümüş tonlu", "a silver-tone jewelry product"),
        ("rose tonlu", "a rose-gold-tone jewelry product"),
        ("siyah tonlu", "a black jewelry product"),
        ("beyaz tonlu", "a white jewelry product"),
        ("çok renkli", "a colorful jewelry product"),
    ],
    "detay": [
        ("taş detaylı", "jewelry with sparkling stones"),
        ("inci detaylı", "jewelry with pearl details"),
        ("boncuk detaylı", "jewelry with bead details"),
        ("zincir detaylı", "jewelry with chain details"),
        ("figürlü", "jewelry with a decorative figure or motif"),
        ("sade tasarımlı", "plain minimal jewelry without stones"),
    ],
    "stil": [
        ("zarif görünümlü", "delicate elegant jewelry"),
        ("modern görünümlü", "modern contemporary jewelry"),
        ("klasik görünümlü", "classic timeless jewelry"),
        ("iddialı görünümlü", "bold statement jewelry"),
    ],
}


def encode_prompts(model, tokenizer, device) -> tuple[np.ndarray, list[tuple[str, str]]]:
    flattened = [
        (group, turkish, prompt)
        for group, choices in ATTRIBUTE_GROUPS.items()
        for turkish, prompt in choices
    ]
    tokens = tokenizer([item[2] for item in flattened]).to(device)
    with torch.inference_mode():
        vectors = model.encode_text(tokens)
        vectors = vectors / vectors.norm(dim=-1, keepdim=True)
    return vectors.float().cpu().numpy(), [(item[0], item[1]) for item in flattened]


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    prompt_vectors, labels = encode_prompts(model, tokenizer, device)

    index = faiss.read_index(str(INDEX_FILE))
    vectors = index.reconstruct_n(0, index.ntotal).astype("float32")
    similarities = vectors @ prompt_vectors.T
    with CODES_FILE.open("r", encoding="utf-8") as handle:
        codes: list[str] = json.load(handle)
    with CATALOG_FILE.open("r", encoding="utf-8") as handle:
        catalog: list[dict] = json.load(handle)
    by_code = {str(item["kod"]): item for item in catalog}

    group_columns: dict[str, list[int]] = {}
    for column, (group, _) in enumerate(labels):
        group_columns.setdefault(group, []).append(column)

    for row, code in enumerate(codes):
        product = by_code[code]
        attributes: dict[str, str] = {}
        confidence: dict[str, float] = {}
        for group, columns in group_columns.items():
            group_scores = similarities[row, columns]
            column = columns[int(group_scores.argmax())]
            attributes[group] = labels[column][1]
            confidence[group] = round(float(similarities[row, column]), 4)
        product["gorsel_ozellikler"] = attributes
        product["gorsel_ozellik_puanlari"] = confidence
        product["aciklama"] = (
            f"{product['kategori']} kategorisinde; {attributes['renk']}, "
            f"{attributes['detay']} ve {attributes['stil']} bir ürün."
        )

    for product in catalog:
        if not product.get("aciklama"):
            product["aciklama"] = f"{product['kategori']} kategorisinde bir ürün."

    backup = CATALOG_FILE.with_name("catalog_before_descriptions.json")
    if not backup.exists():
        shutil.copy2(CATALOG_FILE, backup)
    write_json_atomic(CATALOG_FILE, catalog)
    print(f"Açıklama oluşturuldu: {len(catalog):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Merge code-unique, verified products from the preserved legacy archive."""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np


APP = Path("/var/www/klips-urun-bulucu-v2")
OLD = Path("/var/www/klips-urun-bulucu/urun_bulucu/cache")
CACHE = APP / "cache"
CATALOG_FILE = CACHE / "catalog.json"
CODES_FILE = CACHE / "codes.json"
INDEX_FILE = CACHE / "faiss.index"
IMAGE_DIR = CACHE / "images"
AUDIT_FILE = CACHE / "old_archive_audit.json"


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
    os.replace(temporary, path)


def description(category: str, attributes: dict[str, str]) -> str:
    if not attributes:
        return f"{category} kategorisinde arşivden aktarılan bir ürün."
    return (
        f"{category} kategorisinde; {attributes.get('renk', 'özel tonlu')}, "
        f"{attributes.get('detay', 'dekoratif detaylı')} ve "
        f"{attributes.get('stil', 'şık görünümlü')} bir ürün."
    )


def main() -> int:
    catalog: list[dict] = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    new_codes: list[str] = json.loads(CODES_FILE.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT_FILE.read_text(encoding="utf-8"))
    verified_paths = {item["code"]: Path(item["path"]) for item in audit["valid"]}

    old_codes: list[str] = json.loads((OLD / "codes.json").read_text(encoding="utf-8"))
    old_position = {code: position for position, code in enumerate(old_codes)}
    existing = {str(item["kod"]) for item in catalog}
    candidates = [code for code in old_codes if code not in existing and code in verified_paths]

    if not candidates:
        print("Eklenecek yeni kod yok.")
        return 0

    new_index = faiss.read_index(str(INDEX_FILE))
    old_index = faiss.read_index(str(OLD / "faiss.index"))
    if new_index.d != old_index.d:
        raise RuntimeError("Eski ve yeni embedding boyutları farklı")
    if new_index.ntotal != len(new_codes) or old_index.ntotal != len(old_codes):
        raise RuntimeError("Index ve kod listesi uzunlukları uyuşmuyor")

    rows = np.asarray([old_position[code] for code in candidates], dtype=np.int64)
    all_old_vectors = old_index.reconstruct_n(0, old_index.ntotal).astype("float32")
    candidate_vectors = np.ascontiguousarray(all_old_vectors[rows])

    # Existing hand-labelled products act as a supervised visual reference set.
    neighbors = min(9, new_index.ntotal)
    scores, neighbor_ids = new_index.search(candidate_vectors, neighbors)
    products_by_code = {str(item["kod"]): item for item in catalog}
    reference_categories = [products_by_code[code]["kategori"] for code in new_codes]

    added_products: list[dict] = []
    for row, code in enumerate(candidates):
        votes: dict[str, float] = defaultdict(float)
        for score, neighbor_id in zip(scores[row], neighbor_ids[row]):
            if neighbor_id >= 0:
                votes[reference_categories[int(neighbor_id)]] += max(float(score), 0.0) ** 3
        category = max(votes, key=votes.get) if votes else "DİĞER"
        nearest = products_by_code[new_codes[int(neighbor_ids[row][0])]]
        attributes = dict(nearest.get("gorsel_ozellikler") or {})
        added_products.append(
            {
                "kod": code,
                "urun_adi": f"Arşiv Ürünü {code}",
                "kategori": category,
                "orijinal_kategori": "OTOMATİK GÖRSEL SINIFLANDIRMA",
                "firma": "",
                "resim_url": "",
                "son_gorulme": "Arşivden aktarıldı",
                "gorsel_ozellikler": attributes,
                "gorsel_ozellik_puanlari": {},
                "aciklama": description(category, attributes),
            }
        )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = Path("/var/backups/klips") / f"{stamp}-before-old-archive-merge"
    backup.mkdir(parents=True, exist_ok=False)
    for source in (CATALOG_FILE, CODES_FILE, INDEX_FILE):
        shutil.copy2(source, backup / source.name)

    linked = 0
    for code in candidates:
        source = verified_paths[code]
        destination = IMAGE_DIR / f"{code}.jpg"
        if not destination.exists():
            os.link(source, destination)
            linked += 1

    merged_catalog = catalog + added_products
    merged_codes = new_codes + candidates
    new_index.add(candidate_vectors)
    temporary_index = INDEX_FILE.with_suffix(".index.tmp")
    faiss.write_index(new_index, str(temporary_index))
    atomic_json(CATALOG_FILE, merged_catalog)
    atomic_json(CODES_FILE, merged_codes)
    os.replace(temporary_index, INDEX_FILE)

    print(f"EKLENEN_URUN={len(added_products)}")
    print(f"EKLENEN_GORSEL={linked}")
    print(f"TOPLAM_KATALOG={len(merged_catalog)}")
    print(f"TOPLAM_ARAMA_VEKTORU={new_index.ntotal}")
    print(f"YEDEK={backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CATEGORY_TERMS = [
    (
        "TOKA",
        (
            "hair pin", "hair pins", "hair clip", "hair clips", "barrette",
            "barrettes", "hair accessory", "hair accessories", "scrunchie",
            "scrunchies", "clip", "clips",
        ),
    ),
    ("KÜPE", ("earring", "earrings", "ear plug", "ear plugs")),
    ("KOLYE", ("necklace", "necklaces", "pendant", "pendants", "chain", "chains")),
    ("BİLEKLİK", ("bracelet", "bracelets", "bangle", "bangles")),
    ("YÜZÜK", ("ring", "rings")),
    ("HALHAL", ("anklet",)),
    ("TAÇ", ("headband", "tiara")),
    ("SAAT", ("watch",)),
    ("ÇANTA", ("bag", "purse")),
    ("ANAHTARLIK", ("keychain", "key ring")),
    ("CÜZDAN", ("wallet", "card holder")),
    ("BROŞ", ("brooch", "pin")),
    ("PİERCİNG", ("piercing",)),
    ("KUPA", ("mug", "cup")),
]


def classify(caption: str) -> str:
    text = caption.lower()
    if "belt" in text and "chain" in text:
        return "AKSESUAR"
    hits = []
    for category, terms in CATEGORY_TERMS:
        for term in terms:
            match = re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", text)
            if match:
                hits.append((match.start(), category))
    return min(hits)[1] if hits else "HEDİYELİK"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    captions = {}
    with args.captions.open("r", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            captions[str(item["kod"])] = str(item.get("caption_en", ""))

    products = []
    for code in manifest["new_product_codes"]:
        caption = captions.get(str(code), "")
        category = classify(caption)
        products.append(
            {
                "kod": str(code),
                "urun_adi": f"PDF Ürünü {code}",
                "kategori": category,
                "orijinal_kategori": "PDF İÇE AKTARIM",
                "firma": "PDF RAPOR",
                "resim_url": f"/resim/{code}",
                "son_gorulme": "2026-07-28",
                "aciklama": f"{category.title()} ürünü.",
            }
        )
    args.output.write_text(
        json.dumps(products, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"PRODUCTS={len(products)} OUTPUT={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

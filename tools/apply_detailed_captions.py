"""Convert English Florence captions into conservative Turkish catalog metadata."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path


PRODUCT_WORDS = (
    "necklace", "pendant", "earring", "bracelet", "ring", "anklet", "brooch",
    "jewelry", "jewellery", "chain", "keychain", "hair clip", "hair accessory",
    "book", "notebook", "toy", "tape", "container", "sticker", "glasses",
    "basket", "pen", "scissors", "watch", "bag", "wallet", "headband",
)

OBJECTS = [
    ("çocuk kitabı", ("children's book", "childrens book")),
    ("defter", ("notebook", "spiral book")),
    ("oyuncak", ("toy",)),
    ("dekoratif bant", ("rolls of tape", "package of tape")),
    ("saklama kabı", ("plastic container", "containers")),
    ("çıkartma", ("sticker",)),
    ("parti gözlüğü", ("glasses",)),
    ("sepet", ("basket",)),
    ("kalem", ("a pen", "the pen")),
    ("makas", ("scissors",)),
]

MOTIFS = [
    ("dört yapraklı yonca", ("four-leaf clover", "four leaf clover")),
    ("denizyıldızı", ("starfish", "sea star")),
    ("hayat ağacı", ("tree of life",)),
    ("nazar gözü", ("evil eye",)),
    ("uğur böceği", ("ladybug", "ladybird")),
    ("deniz kabuğu", ("seashell", "sea shell")),
    ("yonca", ("clover",)),
    ("kalp", ("heart",)),
    ("kelebek", ("butterfly",)),
    ("çiçek", ("flower", "floral")),
    ("yaprak", ("leaf", "leaves")),
    ("yıldız", ("star",)),
    ("ay", ("crescent", "moon")),
    ("güneş", ("sun shaped", "sun pendant")),
    ("balık", ("fish",)),
    ("kaplumbağa", ("turtle",)),
    ("fil", ("elephant",)),
    ("yılan", ("snake", "serpent")),
    ("yusufçuk", ("dragonfly",)),
    ("arı", ("bee",)),
    ("kedi", ("cat",)),
    ("pati", ("paw",)),
    ("kuş", ("bird",)),
    ("tüy", ("feather",)),
    ("denizatı", ("seahorse",)),
    ("çapa", ("anchor",)),
    ("anahtar", ("key shaped", "key pendant")),
    ("kilit", ("padlock", "lock pendant")),
    ("sonsuzluk", ("infinity",)),
    ("haç", ("cross shaped", "cross pendant")),
    ("melek", ("angel",)),
    ("taç", ("crown",)),
    ("fiyonk", ("bow shaped", "ribbon bow")),
    ("kiraz", ("cherry",)),
]

METAL_COLORS = [
    ("rose tonlu", ("rose gold", "rose-gold")),
    ("altın tonlu", ("gold colored", "gold-coloured", "gold in color", "gold-tone", "golden", "gold")),
    ("gümüş tonlu", ("silver colored", "silver-coloured", "silver in color", "silver-tone", "silver")),
    ("çok renkli", ("multicolored", "multi-colored", "colorful")),
]

ACCENT_COLORS = [
    ("siyah", ("black",)),
    ("yeşil", ("green",)),
    ("mavi", ("blue",)),
    ("turkuaz", ("turquoise",)),
    ("kırmızı", ("red",)),
    ("pembe", ("pink",)),
    ("mor", ("purple",)),
]

DETAILS = [
    ("inci detaylı", ("pearl",)),
    ("boncuk detaylı", ("bead", "beaded")),
    ("mine detaylı", ("enamel",)),
    ("taş detaylı", ("rhinestone", "crystal", "gemstone", "sparkling stone", "small stones")),
]

FORMS = [
    ("halka formunda", ("hoop",)),
    ("sallantılı", ("dangling", "drop earring")),
    ("katmanlı", ("layered", "multiple chains")),
    ("kalın formlu", ("chunky", "thick chain")),
]


def atomic_json(path: Path, value: object) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
        temporary = Path(stream.name)
    os.replace(temporary, path)


def relevant_caption(caption: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", caption.lower())
    relevant = [sentence for sentence in sentences if any(word in sentence for word in PRODUCT_WORDS)]
    return " ".join(relevant)


def attribute_caption(caption: str) -> str:
    text = relevant_caption(caption)
    return re.sub(
        r"\b(?:black|white|pink|red|gray|grey|dark)\s+"
        r"(?:tag|card|background|surface|display|packaging)\b",
        " ",
        text,
    )


def first_match(text: str, choices: list[tuple[str, tuple[str, ...]]]) -> str:
    for label, aliases in choices:
        if any(alias in text for alias in aliases):
            return label
    return ""


def all_matches(text: str, choices: list[tuple[str, tuple[str, ...]]], limit: int = 2) -> list[str]:
    return [label for label, aliases in choices if any(alias in text for alias in aliases)][:limit]


def build_description(
    product: dict, color: str, accent: str, motif: str, details: list[str], form: str,
    object_name: str = "",
) -> str:
    parts = [color]
    if accent and accent not in color:
        parts.append(f"{accent} renkli")
    if motif:
        parts.append(f"{motif} figürlü")
    parts.extend(details)
    if form:
        parts.append(form)
    category = object_name or str(product["kategori"]).translate(
        str.maketrans({"I": "ı", "İ": "i"})
    ).lower()
    return f"{', '.join(parts).capitalize()} {category}."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    captions: dict[str, str] = {}
    with args.captions.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                item = json.loads(line)
                if item.get("caption_en"):
                    caption = re.sub(r"<[^>]+>", " ", str(item["caption_en"]))
                    captions[str(item["kod"])] = re.sub(r"\s+", " ", caption).strip()
            except json.JSONDecodeError:
                continue

    catalog: list[dict] = json.loads(args.catalog.read_text(encoding="utf-8"))
    enriched = 0
    motifs = 0
    for product in catalog:
        code = str(product["kod"])
        caption = captions.get(code, "")
        if not caption:
            continue
        relevant = attribute_caption(caption)
        existing = product.get("gorsel_ozellikler") or {}
        color = first_match(relevant, METAL_COLORS) or str(existing.get("renk") or "özel tonlu")
        accent = first_match(relevant, ACCENT_COLORS)
        motif = first_match(relevant, MOTIFS)
        details = all_matches(relevant, DETAILS)
        form = first_match(relevant, FORMS)
        object_name = first_match(relevant, OBJECTS)
        product["gorsel_ozellikler_detayli"] = {
            "renk": color, "renk_detayi": accent, "motif": motif,
            "detaylar": details, "form": form, "obje": object_name,
        }
        product["arama_etiketleri"] = [
            value
            for value in [product["kategori"], color, accent, motif, *details, form, object_name]
            if value
        ]
        product["aciklama"] = build_description(
            product, color, accent, motif, details, form, object_name
        )
        enriched += 1
        motifs += bool(motif)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, catalog)
    print(f"CATALOG={len(catalog)} ENRICHED={enriched} MOTIF_ASSIGNED={motifs} OUTPUT={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

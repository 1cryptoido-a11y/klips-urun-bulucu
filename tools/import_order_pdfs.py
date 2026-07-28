from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pdfplumber
from PIL import Image, ImageFilter, ImageStat
from pypdf import PdfReader


BARCODE_PATTERN = re.compile(r"^\d{6}$")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_score(image: Image.Image) -> float:
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    sharpness = ImageStat.Stat(edges).var[0]
    return float(image.width * image.height) + sharpness * 100.0


def barcode_words(page: pdfplumber.page.Page) -> list[dict]:
    result = []
    for word in page.extract_words():
        text = word.get("text", "").strip()
        if BARCODE_PATTERN.fullmatch(text):
            result.append(word)
    return result


def match_rows(page: pdfplumber.page.Page) -> list[tuple[str, dict]]:
    codes = barcode_words(page)
    images = [
        image
        for image in page.images
        if image.get("width", 0) >= 50
        and image.get("height", 0) >= 50
        and image.get("top", 0) >= 100
    ]
    matches: list[tuple[str, dict]] = []
    used_names: set[str] = set()
    for code in codes:
        code_y = (code["top"] + code["bottom"]) / 2
        candidates = []
        for image in images:
            name = str(image.get("name", ""))
            if not name or name in used_names:
                continue
            image_y = (image["top"] + image["bottom"]) / 2
            distance = abs(image_y - code_y)
            tolerance = max(55.0, image.get("height", 0) / 2 + 15.0)
            if distance <= tolerance:
                candidates.append((distance, image))
        if candidates:
            _, selected = min(candidates, key=lambda item: item[0])
            used_names.add(str(selected["name"]))
            matches.append((code["text"], selected))
    return matches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    catalog_codes = {str(product.get("kod", "")).strip() for product in catalog}
    seen_pdf_hashes: dict[str, str] = {}
    occurrences: dict[str, list[dict]] = defaultdict(list)
    best_images: dict[str, tuple[float, Image.Image]] = {}
    skipped_duplicates = []
    errors = []
    page_count = 0

    for pdf_path in sorted(args.pdf_dir.glob("*.pdf")):
        try:
            digest = file_hash(pdf_path)
            if digest in seen_pdf_hashes:
                skipped_duplicates.append(
                    {"file": pdf_path.name, "same_as": seen_pdf_hashes[digest]}
                )
                continue
            seen_pdf_hashes[digest] = pdf_path.name
            reader = PdfReader(pdf_path)
            with pdfplumber.open(pdf_path) as plumber_pdf:
                for page_index, (page, reader_page) in enumerate(
                    zip(plumber_pdf.pages, reader.pages), start=1
                ):
                    page_count += 1
                    rows = match_rows(page)
                    embedded = {
                        Path(item.name).stem: item.image.convert("RGB")
                        for item in reader_page.images
                    }
                    for code, image_info in rows:
                        image = embedded.get(str(image_info.get("name", "")))
                        if image is None:
                            continue
                        occurrences[code].append(
                            {"pdf": pdf_path.name, "page": page_index}
                        )
                        score = image_score(image)
                        if code not in best_images or score > best_images[code][0]:
                            best_images[code] = (score, image.copy())
        except Exception as exc:
            errors.append({"file": pdf_path.name, "error": str(exc)})

    for code, (_, image) in best_images.items():
        image.save(image_dir / f"{code}.jpg", "JPEG", quality=95, optimize=True)

    codes = set(best_images)
    manifest = {
        "pdf_files": len(seen_pdf_hashes),
        "duplicate_pdf_files": skipped_duplicates,
        "pages_scanned": page_count,
        "products_with_images": len(codes),
        "new_product_codes": sorted(codes - catalog_codes),
        "existing_product_codes": sorted(codes & catalog_codes),
        "catalog_codes_without_pdf_match": len(catalog_codes - codes),
        "occurrence_counts": dict(Counter(len(value) for value in occurrences.values())),
        "occurrences": occurrences,
        "errors": errors,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"PDF={manifest['pdf_files']} PAGES={page_count} "
        f"PRODUCTS={len(codes)} NEW={len(manifest['new_product_codes'])} "
        f"EXISTING={len(manifest['existing_product_codes'])} "
        f"DUPLICATE_PDF={len(skipped_duplicates)} ERRORS={len(errors)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Merge a browser-exported report snapshot and download its new product images."""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, ImageOps

from import_catalog import infer_category, normalize_text


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def download(product: dict, image_dir: Path) -> tuple[str, str]:
    code = str(product["kod"])
    target = image_dir / f"{code}.jpg"
    if target.is_file():
        try:
            with Image.open(target) as image:
                image.verify()
            return code, "existing"
        except Exception:
            target.unlink(missing_ok=True)
    url = str(product.get("resim_url") or "")
    if not url:
        return code, "missing_url"
    try:
        response = requests.get(url, timeout=(10, 45), headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        if len(response.content) < 512:
            raise ValueError("image response is too small")
        with Image.open(io.BytesIO(response.content)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            if image.width < 40 or image.height < 40:
                raise ValueError(f"image is too small: {image.size}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp")
            image.save(temporary, format="JPEG", quality=92, optimize=True)
            os.replace(temporary, target)
        return code, "downloaded"
    except Exception as exc:
        return code, f"error: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--base-catalog", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-catalog", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    base: list[dict] = json.loads(args.base_catalog.read_text(encoding="utf-8"))
    snapshot: list[dict] = json.loads(args.snapshot.read_text(encoding="utf-8"))
    existing = {str(item["kod"]) for item in base}
    additions: list[dict] = []
    for row in snapshot:
        code = normalize_text(row.get("kod"))
        if not code or code in existing:
            continue
        name = normalize_text(row.get("urun_adi"))
        additions.append({
            "kod": code,
            "urun_adi": name,
            "kategori": infer_category(code, name, row.get("kategori")),
            "orijinal_kategori": normalize_text(row.get("kategori")),
            "firma": normalize_text(row.get("firma")),
            "resim_url": normalize_text(row.get("resim_url")),
            "son_gorulme": "2026-07-28",
            "aciklama": f"{infer_category(code, name, row.get('kategori')).title()} ürünü.",
            "arama_etiketleri": [infer_category(code, name, row.get("kategori"))],
        })
        existing.add(code)

    args.image_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, item, args.image_dir): item["kod"] for item in additions}
        for number, future in enumerate(as_completed(futures), 1):
            code, status = future.result()
            results[code] = status
            if number % 250 == 0 or number == len(futures):
                ok = sum(value in {"downloaded", "existing"} for value in results.values())
                print(f"CHECKED={number}/{len(futures)} VALID={ok}", flush=True)

    merged = base + additions
    atomic_json(args.output_catalog, merged)
    report = {
        "snapshot_rows": len(snapshot), "base_products": len(base),
        "new_products": len(additions), "final_products": len(merged),
        "valid_images": sum(value in {"downloaded", "existing"} for value in results.values()),
        "failed_images": {code: status for code, status in results.items() if status not in {"downloaded", "existing"}},
    }
    atomic_json(args.report, report)
    print(json.dumps({key: value for key, value in report.items() if key != "failed_images"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

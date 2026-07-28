"""Import and download the Klips product catalog from the franchise report.

The authenticated report URL is read from ``KLIPS_REPORT_URL`` so credentials
and access tokens never need to be stored in this project.
"""

from __future__ import annotations

import argparse
import calendar
import json
import logging
import os
import shutil
import tempfile
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

import requests


LOGGER = logging.getLogger("catalog_import")
DATA_ENDPOINT = "https://karanfil.satis.web.tr/rapor/rapor_v2_data_v2"


CATEGORY_ALIASES = {
    "BILEKLIK": "BİLEKLİK",
    "BİLEKLİK": "BİLEKLİK",
    "BROS": "BROŞ",
    "BROŞ": "BROŞ",
    "HEDIYELIK": "HEDİYELİK",
    "HEDİYELİK": "HEDİYELİK",
    "PIERCING": "PİERCİNG",
    "PİERCİNG": "PİERCİNG",
    "SAHMARAN": "ŞAHMARAN",
    "ŞAHMARAN": "ŞAHMARAN",
    "SARF MALZEME": "SARF MALZEME",
    "TOKA": "TOKA",
}

CATEGORY_OVERRIDES = {
    "004453": "YÜZÜK",  # verified from the product image
    "100000": "DİĞER",
    "127921": "DİĞER",
}


@dataclass(frozen=True)
class MonthRange:
    start: date
    end: date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2025-01-01", type=date.fromisoformat)
    parser.add_argument("--end", default=date.today().isoformat(), type=date.fromisoformat)
    parser.add_argument("--output-dir", type=Path, default=Path("cache"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def month_ranges(start: date, end: date) -> Iterator[MonthRange]:
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        last_day = calendar.monthrange(cursor.year, cursor.month)[1]
        month_end = date(cursor.year, cursor.month, last_day)
        yield MonthRange(max(start, cursor), min(end, month_end))
        cursor = date(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1)


def normalize_text(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def normalize_category(value: Any) -> str:
    category = " ".join(normalize_text(value).split()).upper()
    return CATEGORY_ALIASES.get(category, category)


def infer_category(barcode: str, product_name: str, raw_category: Any) -> str:
    category = normalize_category(raw_category)
    if category:
        return category
    if barcode in CATEGORY_OVERRIDES:
        return CATEGORY_OVERRIDES[barcode]
    name = product_name.upper()
    rules = (
        (("-KLY", "KOLYE"), "KOLYE"),
        (("-ŞHM", "-SHM"), "ŞAHMARAN"),
        (("-KP",), "KÜPE"),
        (("-ÇNT", "-CNT"), "ÇANTA"),
        (("OYUN",), "HEDİYELİK"),
        (("POŞET", "POSET"), "SARF MALZEME"),
    )
    for markers, result in rules:
        if any(marker in name for marker in markers):
            return result
    return "DİĞER"


def create_session(report_url: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "KlipsCatalogImporter/1.0",
            "Referer": report_url,
            "X-Requested-With": "XMLHttpRequest",
        }
    )
    response = session.get(report_url, timeout=30)
    response.raise_for_status()
    return session


def fetch_month(session: requests.Session, period: MonthRange) -> list[dict[str, Any]]:
    response = session.post(
        DATA_ENDPOINT,
        data={"sartdate": period.start.isoformat(), "enddate": period.end.isoformat()},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    rows = payload.get("rows") or []
    if not isinstance(rows, list):
        raise TypeError("Report response does not contain a row list")
    return rows


def product_from_row(row: dict[str, Any], seen_at: date) -> dict[str, Any] | None:
    barcode = normalize_text(row.get("barkod") or row.get("stok_kodu"))
    if not barcode:
        return None
    product_name = normalize_text(row.get("stok_ismi"))
    return {
        "kod": barcode,
        "urun_adi": product_name,
        "kategori": infer_category(barcode, product_name, row.get("grup")),
        "orijinal_kategori": normalize_text(row.get("grup")),
        "firma": normalize_text(row.get("firma")),
        "resim_url": normalize_text(row.get("image") or row.get("urun_gorsel")),
        "son_gorulme": seen_at.isoformat(),
    }


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp"
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def collect_catalog(
    session: requests.Session, start: date, end: date, output_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    products: dict[str, dict[str, Any]] = {}
    month_stats: list[dict[str, Any]] = []

    for period in month_ranges(start, end):
        rows = fetch_month(session, period)
        before = len(products)
        for row in rows:
            product = product_from_row(row, period.end)
            if product:
                products[product["kod"]] = product
        added = len(products) - before
        month_stats.append(
            {
                "baslangic": period.start.isoformat(),
                "bitis": period.end.isoformat(),
                "satir": len(rows),
                "yeni_barkod": added,
            }
        )
        LOGGER.info(
            "%s - %s: %s rows, %s new barcodes, %s total",
            period.start,
            period.end,
            f"{len(rows):,}",
            f"{added:,}",
            f"{len(products):,}",
        )

        checkpoint = sorted(products.values(), key=lambda item: item["kod"])
        write_json_atomic(output_dir / "catalog_checkpoint.json", checkpoint)

    catalog = sorted(products.values(), key=lambda item: item["kod"])
    categories = Counter(item["kategori"] for item in catalog)
    report = {
        "olusturma_zamani": datetime.now().astimezone().isoformat(),
        "baslangic": start.isoformat(),
        "bitis": end.isoformat(),
        "benzersiz_urun": len(catalog),
        "resim_adresi_olan": sum(bool(item["resim_url"]) for item in catalog),
        "kategoriler": dict(categories.most_common()),
        "aylar": month_stats,
    }
    write_json_atomic(output_dir / "catalog.json", catalog)
    write_json_atomic(output_dir / "catalog_report.json", report)
    return catalog, report


def download_one(session: requests.Session, product: dict[str, Any], image_dir: Path) -> str:
    destination = image_dir / f"{product['kod']}.jpg"
    if destination.exists() and destination.stat().st_size > 0:
        return "existing"

    image_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        normalize_text(product.get("resim_url")),
        f"https://resim.satis.web.tr/rapor/{product['kod']}.jpg",
        f"https://resim.satis.web.tr/merkez/{product['kod']}.jpg",
    ]
    for url in dict.fromkeys(candidate for candidate in candidates if candidate):
        response = session.get(url, stream=True, timeout=45)
        if response.status_code == 404:
            continue
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if content_type and not content_type.startswith("image/"):
            continue

        with tempfile.NamedTemporaryFile(delete=False, dir=image_dir, suffix=".part") as handle:
            temp_path = Path(handle.name)
            shutil.copyfileobj(response.raw, handle)
        if temp_path.stat().st_size == 0:
            temp_path.unlink(missing_ok=True)
            continue
        temp_path.replace(destination)
        product["resim_url"] = url
        return "downloaded"
    return "not_found"


def download_images(
    session: requests.Session,
    catalog: list[dict[str, Any]],
    output_dir: Path,
    workers: int,
    limit: int | None,
) -> dict[str, int]:
    selected = catalog[:limit] if limit else catalog
    image_dir = output_dir / "images"
    failures: list[dict[str, str]] = []
    results: Counter[str] = Counter()

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(download_one, session, product, image_dir): product
            for product in selected
        }
        for completed, future in enumerate(as_completed(futures), 1):
            product = futures[future]
            try:
                status = future.result()
            except Exception as exc:  # keep the full import running
                status = "error"
                failures.append({"kod": product["kod"], "hata": str(exc)})
            else:
                if status not in {"downloaded", "existing"}:
                    failures.append({"kod": product["kod"], "hata": status})
            results[status] += 1
            if completed % 250 == 0 or completed == len(selected):
                LOGGER.info("Images: %s / %s", f"{completed:,}", f"{len(selected):,}")

    write_json_atomic(output_dir / "image_failures.json", failures)
    write_json_atomic(output_dir / "image_report.json", dict(results))
    return dict(results)


def main() -> int:
    args = parse_args()
    if args.start > args.end:
        raise SystemExit("--start cannot be after --end")
    report_url = os.environ.get("KLIPS_REPORT_URL", "").strip()
    if not report_url:
        raise SystemExit("KLIPS_REPORT_URL is required")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    session = create_session(report_url)
    catalog, report = collect_catalog(session, args.start, args.end, args.output_dir)
    LOGGER.info(
        "Catalog complete: %s unique products", f"{report['benzersiz_urun']:,}"
    )

    if args.download:
        image_report = download_images(
            session, catalog, args.output_dir, args.workers, args.limit
        )
        LOGGER.info("Image results: %s", image_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

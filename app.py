from __future__ import annotations

import json
import logging
import tempfile
import base64
from io import BytesIO
from pathlib import Path

from flask import Flask, abort, render_template, request, send_file
from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener

from ai.search_engine import get_engine
from ai.query_processor import normalize_uploaded_image
from config import CATALOG_FILE, IMAGE_DIR, MAX_UPLOAD_BYTES, RESULT_COUNT


register_heif_opener()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_categories() -> list[str]:
    if not CATALOG_FILE.exists():
        return []
    with CATALOG_FILE.open("r", encoding="utf-8") as handle:
        products = json.load(handle)
    return sorted({p.get("kategori", "") for p in products if p.get("kategori")})


def make_query_preview(path: str | Path) -> str:
    """Create a compact browser-safe preview that survives the result response."""
    with Image.open(path) as opened:
        preview = opened.convert("RGB")
    preview.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    preview.save(buffer, "JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    @app.get("/")
    def home():
        return render_template("index.html", categories=load_categories())

    @app.post("/ara")
    def search():
        photo = request.files.get("foto")
        description = request.form.get("aciklama", "").strip()
        category = request.form.get("kategori", "").strip()
        if (not photo or not photo.filename) and not description:
            return render_template(
                "index.html",
                categories=load_categories(),
                error="Fotoğraf yükleyin veya bir açıklama yazın.",
            ), 400

        try:
            with tempfile.TemporaryDirectory(prefix="klips-search-") as temp_dir:
                image_path: Path | None = None
                query_preview: str | None = None
                if photo and photo.filename:
                    original = Path(temp_dir) / "query"
                    photo.save(original)
                    image_path = normalize_uploaded_image(
                        original, Path(temp_dir) / "query.jpg"
                    )
                    query_preview = make_query_preview(image_path)
                results = get_engine().search(
                    image_path=image_path,
                    description=description,
                    category=category,
                    limit=RESULT_COUNT,
                )
        except (UnidentifiedImageError, OSError):
            return render_template(
                "index.html", categories=load_categories(), error="Görsel okunamadı."
            ), 400
        except Exception:
            app.logger.exception("Search failed")
            return render_template(
                "index.html",
                categories=load_categories(),
                error="Arama sırasında bir hata oluştu.",
            ), 500

        return render_template(
            "index.html",
            categories=load_categories(),
            results=results,
            selected_category=category,
            description=description,
            query_preview=query_preview,
        )

    @app.post("/kod")
    def search_code():
        code = request.form.get("kod", "").strip()
        product = get_engine().find_by_code(code) if code else None
        return render_template(
            "index.html",
            categories=load_categories(),
            code_query=code,
            results=[{**product, "puan": 1.0}] if product else [],
            error=None if product else "Bu ürün kodu bulunamadı.",
        ), 200 if product else 404

    @app.get("/resim/<code>")
    def product_image(code: str):
        if not code or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in code):
            abort(404)
        path = IMAGE_DIR / f"{code}.jpg"
        if not path.is_file():
            abort(404)
        return send_file(path, conditional=True, max_age=86400)

    @app.get("/health")
    def health():
        return {"status": "ok", "catalog": CATALOG_FILE.exists(), "index": (CATALOG_FILE.parent / "faiss.index").exists()}

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)

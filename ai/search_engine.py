from __future__ import annotations

import json
import re
import threading
import unicodedata
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import open_clip
import torch
from PIL import Image

from config import (
    CATALOG_FILE,
    CATEGORY_SEARCH_CANDIDATES,
    CODES_FILE,
    IMAGE_WEIGHT,
    INDEX_FILE,
    MODEL_NAME,
    FOCUSED_VIEW_WEIGHT,
    GRAYSCALE_INDEX_FILE,
    PRETRAINED,
    SEARCH_CANDIDATES,
    TEXT_WEIGHT,
)
from ai.query_processor import prepare_query_views


STOP_WORDS = {
    "bir", "ve", "ile", "olan", "gibi", "ürün", "urün", "tonlu", "renkli",
    "detaylı", "detayli", "figürlü", "figurlu", "model", "tasarım",
}


def search_terms(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return {
        token for token in re.findall(r"[a-zçğıöşü0-9]+", normalized)
        if len(token) > 1 and token not in STOP_WORDS
    }


def lexical_score(description: str, product: dict[str, Any]) -> float:
    query = search_terms(description)
    if not query:
        return 0.0
    metadata = " ".join(
        [
            str(product.get("kategori", "")),
            str(product.get("aciklama", "")),
            " ".join(str(item) for item in product.get("arama_etiketleri", [])),
        ]
    )
    overlap = query & search_terms(metadata)
    return len(overlap) / len(query)


def text_only_score(description: str, product: dict[str, Any], clip_score: float) -> float:
    """Prefer exact Turkish catalog metadata; use CLIP only as a tie-breaker."""
    lexical = lexical_score(description, product)
    return (0.85 * lexical) + (0.15 * max(clip_score, 0.0))


class ProductSearchEngine:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED
        )
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        self.model = self.model.to(self.device).eval()
        self.index = faiss.read_index(str(INDEX_FILE))
        self.grayscale_index = (
            faiss.read_index(str(GRAYSCALE_INDEX_FILE))
            if GRAYSCALE_INDEX_FILE.exists()
            else None
        )
        with CODES_FILE.open("r", encoding="utf-8") as handle:
            self.codes: list[str] = json.load(handle)
        with CATALOG_FILE.open("r", encoding="utf-8") as handle:
            products = json.load(handle)
        self.products: dict[str, dict[str, Any]] = {
            str(product["kod"]): product for product in products
        }
        if self.index.ntotal != len(self.codes):
            raise RuntimeError("FAISS index and code list have different lengths")
        if self.grayscale_index is not None and self.grayscale_index.ntotal != len(self.codes):
            raise RuntimeError("Grayscale index and code list have different lengths")
        self.category_indexes: dict[str, tuple[faiss.IndexFlatIP, np.ndarray]] = {}
        vectors = self.index.reconstruct_n(0, self.index.ntotal).astype("float32")
        category_rows: dict[str, list[int]] = {}
        for row, code in enumerate(self.codes):
            category = str(self.products.get(code, {}).get("kategori", ""))
            if category:
                category_rows.setdefault(category, []).append(row)
        for category, rows in category_rows.items():
            row_ids = np.asarray(rows, dtype=np.int64)
            category_index = faiss.IndexFlatIP(self.index.d)
            category_index.add(np.ascontiguousarray(vectors[row_ids]))
            self.category_indexes[category] = (category_index, row_ids)

    def _encode_images(self, image_path: str | Path, category: str = "") -> torch.Tensor:
        views = prepare_query_views(image_path, category)
        tensor = torch.stack([self.preprocess(view) for view in views]).to(self.device)
        with torch.inference_mode():
            vectors = self.model.encode_image(tensor)
        return vectors / vectors.norm(dim=-1, keepdim=True)

    def _encode_text(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer([text]).to(self.device)
        with torch.inference_mode():
            vector = self.model.encode_text(tokens)
        return vector / vector.norm(dim=-1, keepdim=True)

    def _query_vectors(
        self, image_path: str | Path | None, description: str, category: str = ""
    ) -> np.ndarray:
        image_vectors = self._encode_images(image_path, category) if image_path else None
        text_vector = self._encode_text(description) if description else None
        if image_vectors is None and text_vector is None:
            raise ValueError("A photo or description is required")
        if image_vectors is not None and text_vector is not None:
            text_vectors = text_vector.expand(image_vectors.shape[0], -1)
            vectors = IMAGE_WEIGHT * image_vectors + TEXT_WEIGHT * text_vectors
            vectors = vectors / vectors.norm(dim=-1, keepdim=True)
        elif image_vectors is not None:
            vectors = image_vectors
        else:
            vectors = text_vector
        return vectors.detach().cpu().numpy().astype("float32")

    def search(
        self,
        *,
        image_path: str | Path | None = None,
        description: str = "",
        category: str = "",
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        queries = self._query_vectors(image_path, description.strip(), category)
        grayscale_queries: np.ndarray | None = None
        if image_path:
            color_view_count = len(queries) // 2
            grayscale_queries = queries[color_view_count:]
            queries = queries[:color_view_count]
        search_index = self.index
        row_ids: np.ndarray | None = None
        if category and category in self.category_indexes:
            search_index, row_ids = self.category_indexes[category]
        requested = CATEGORY_SEARCH_CANDIDATES if category else SEARCH_CANDIDATES
        requested = min(search_index.ntotal, max(requested, limit * 20))
        scores, ids = search_index.search(queries, requested)

        # Fuse original and automatically focused views, retaining the best
        # evidence for each product before applying the category constraint.
        fused: dict[int, float] = {}
        for view, (view_scores, view_ids) in enumerate(zip(scores, ids)):
            weight = 1.0 if view == 0 else FOCUSED_VIEW_WEIGHT
            if category.upper() == "KOLYE" and view == len(scores) - 1:
                # The final necklace view is a deliberately tight pendant crop.
                # On a full display-card photo it carries more product identity
                # than the card, logo, hand, and long chain in the original.
                weight = 1.10
            for item_id, score in zip(view_ids, view_scores):
                if item_id < 0:
                    continue
                if row_ids is not None:
                    item_id = int(row_ids[item_id])
                weighted = float(score) * weight
                fused[item_id] = max(fused.get(item_id, -1.0), weighted)

        if grayscale_queries is not None and self.grayscale_index is not None:
            grayscale_requested = min(
                self.grayscale_index.ntotal,
                max(SEARCH_CANDIDATES, CATEGORY_SEARCH_CANDIDATES * 3 if category else 0),
            )
            grayscale_scores, grayscale_ids = self.grayscale_index.search(
                grayscale_queries, grayscale_requested
            )
            for view_scores, view_ids in zip(grayscale_scores, grayscale_ids):
                for item_id, score in zip(view_ids, view_scores):
                    if item_id < 0:
                        continue
                    weighted = float(score) * FOCUSED_VIEW_WEIGHT
                    fused[item_id] = max(fused.get(item_id, -1.0), weighted)

        ranked: list[tuple[float, dict[str, Any]]] = []
        candidate_ids = set(fused)
        if description and image_path is None:
            # Turkish descriptions and motif names are substantially more reliable
            # in catalog metadata than in the English-heavy CLIP text encoder. Add
            # every indexed lexical match so it cannot be excluded by FAISS recall.
            for item_id, code in enumerate(self.codes):
                product = self.products.get(code)
                if product and (not category or product.get("kategori") == category):
                    if lexical_score(description, product) > 0:
                        candidate_ids.add(item_id)

        for item_id in candidate_ids:
            score = fused.get(item_id, 0.0)
            code = self.codes[item_id]
            product = self.products.get(code)
            if not product or (category and product.get("kategori") != category):
                continue
            if description and image_path is None:
                adjusted = text_only_score(description, product, score)
            else:
                adjusted = score + (0.045 * lexical_score(description, product) if description else 0.0)
            ranked.append((adjusted, product))

        results: list[dict[str, Any]] = []
        for score, product in sorted(ranked, key=lambda item: item[0], reverse=True):
            results.append({**product, "puan": min(float(score), 1.0)})
            if len(results) >= limit:
                break
        return results

    def find_by_code(self, code: str) -> dict[str, Any] | None:
        return self.products.get(code.strip())

    def categories(self) -> list[str]:
        return sorted({p.get("kategori", "") for p in self.products.values() if p.get("kategori")})


_engine: ProductSearchEngine | None = None
_lock = threading.Lock()


def get_engine() -> ProductSearchEngine:
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = ProductSearchEngine()
    return _engine

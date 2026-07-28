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
from PIL import Image, ImageOps
from scipy import ndimage

from config import (
    CATALOG_FILE,
    CATEGORY_SEARCH_CANDIDATES,
    CODES_FILE,
    DINO_INDEX_FILE,
    IMAGE_DIR,
    IMAGE_WEIGHT,
    INDEX_FILE,
    MODEL_NAME,
    OBJECT_INDEX_FILE,
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


def select_necklace_dino_queries(
    queries: np.ndarray,
    *,
    macro_necklace: bool,
    layered_necklace: bool,
) -> np.ndarray | None:
    """Route DINO evidence without letting a tight pendant crop hide the model."""
    if macro_necklace:
        return queries
    if not layered_necklace:
        return None
    if len(queries) >= 6:
        return queries[4:6]
    return queries[:1]


def count_blue_motif_centers(image: Image.Image) -> int:
    """Count meaningful blue/cyan motif centers, ignoring their exact shade."""
    rgb = np.asarray(image.convert("RGB").resize((512, 512)), dtype="float32")
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (blue > red * 1.08) & (blue > green * 1.03) & ((blue - red) > 10) & (blue > 60)
    labels, count = ndimage.label(mask)
    if not count:
        return 0
    areas = np.bincount(labels.ravel())[1:]
    return int(np.count_nonzero((areas >= 60) & (areas <= 1500)))


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
        necklace_prompts = self.tokenizer(
            [
                "a layered multi strand necklace with several chains and pendants",
                "a single pendant necklace on a display card",
            ]
        ).to(self.device)
        with torch.inference_mode():
            necklace_text_vectors = self.model.encode_text(necklace_prompts)
        necklace_text_vectors = necklace_text_vectors / necklace_text_vectors.norm(
            dim=-1, keepdim=True
        )
        self.necklace_type_vectors = (
            necklace_text_vectors.float().cpu().numpy().astype("float32")
        )
        self.visual_category_labels = ("KOLYE", "KÜPE", "YÜZÜK", "BİLEKLİK", "TOKA")
        category_prompts = self.tokenizer(
            [
                "a necklace jewelry product",
                "a pair of earrings jewelry product",
                "a ring jewelry product",
                "a bracelet jewelry product",
                "a hair clip accessory",
            ]
        ).to(self.device)
        with torch.inference_mode():
            category_text_vectors = self.model.encode_text(category_prompts)
        category_text_vectors = category_text_vectors / category_text_vectors.norm(
            dim=-1, keepdim=True
        )
        self.visual_category_vectors = (
            category_text_vectors.float().cpu().numpy().astype("float32")
        )
        self.visual_motif_labels = (
            "nazar gözü",
            "yonca",
            "kalp",
            "yıldız",
            "kelebek",
            "çiçek",
            "inci",
        )
        motif_prompts = self.tokenizer(
            [
                "a jewelry product with a row of evil eye beads",
                "a jewelry product with a clover charm",
                "a jewelry product with a heart charm",
                "a jewelry product with a star charm",
                "a jewelry product with a butterfly charm",
                "a jewelry product with a flower charm",
                "a pearl jewelry product",
            ]
        ).to(self.device)
        with torch.inference_mode():
            motif_text_vectors = self.model.encode_text(motif_prompts)
        motif_text_vectors = motif_text_vectors / motif_text_vectors.norm(
            dim=-1, keepdim=True
        )
        self.visual_motif_vectors = (
            motif_text_vectors.float().cpu().numpy().astype("float32")
        )
        self.index = faiss.read_index(str(INDEX_FILE))
        self.grayscale_index = (
            faiss.read_index(str(GRAYSCALE_INDEX_FILE))
            if GRAYSCALE_INDEX_FILE.exists()
            else None
        )
        self.object_index = (
            faiss.read_index(str(OBJECT_INDEX_FILE))
            if OBJECT_INDEX_FILE.exists()
            else None
        )
        self.dino_index = (
            faiss.read_index(str(DINO_INDEX_FILE))
            if DINO_INDEX_FILE.exists()
            else None
        )
        self.dino_model = None
        self.dino_transform = None
        if self.dino_index is not None:
            import timm
            from timm.data import create_transform, resolve_model_data_config

            self.dino_model = timm.create_model(
                "vit_small_patch14_dinov2.lvd142m",
                pretrained=True,
                num_classes=0,
            ).to(self.device).eval()
            self.dino_transform = create_transform(
                **resolve_model_data_config(self.dino_model),
                is_training=False,
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
        if self.object_index is not None and self.object_index.ntotal != len(self.codes):
            raise RuntimeError("Object index and code list have different lengths")
        if self.dino_index is not None and self.dino_index.ntotal != len(self.codes):
            raise RuntimeError("DINOv2 index and code list have different lengths")
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
        if category.upper() == "BİLEKLİK" and len(views) >= 6:
            color_count = len(views) // 2
            selected = [0, 3, color_count - 2, color_count - 1]
            views = [views[index] for index in selected] + [
                views[color_count + index] for index in selected
            ]
        tensor = torch.stack([self.preprocess(view) for view in views]).to(self.device)
        with torch.inference_mode():
            vectors = self.model.encode_image(tensor)
        return vectors / vectors.norm(dim=-1, keepdim=True)

    def _encode_text(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer([text]).to(self.device)
        with torch.inference_mode():
            vector = self.model.encode_text(tokens)
        return vector / vector.norm(dim=-1, keepdim=True)

    def _infer_visual_category(self, image_path: str | Path) -> str:
        """Choose a main product route when the user leaves category empty."""
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        # Display cards and their logos can dominate a thin bracelet. Include
        # product-heavy lower bands and retain the strongest category evidence.
        views = [
            image,
            image.crop((0, int(height * 0.40), width, height)),
            image.crop((0, int(height * 0.55), width, height)),
        ]
        tensor = torch.stack([self.preprocess(view) for view in views]).to(self.device)
        with torch.inference_mode():
            vectors = self.model.encode_image(tensor)
        vectors = vectors / vectors.norm(dim=-1, keepdim=True)
        view_scores = (
            vectors.float().cpu().numpy() @ self.visual_category_vectors.T
        )
        scores = view_scores.max(axis=0)
        return self.visual_category_labels[int(np.argmax(scores))]

    def _encode_dino_images(
        self, image_path: str | Path, category: str
    ) -> np.ndarray | None:
        if self.dino_model is None or self.dino_transform is None:
            return None
        views = prepare_query_views(image_path, category)
        color_views = views[: len(views) // 2]
        if category.upper() == "BİLEKLİK" and len(color_views) >= 3:
            color_views = [color_views[0], color_views[3], color_views[-2], color_views[-1]]
        tensor = torch.stack([self.dino_transform(view) for view in color_views]).to(
            self.device
        )
        with torch.inference_mode():
            vectors = self.dino_model(tensor)
        vectors = vectors / vectors.norm(dim=-1, keepdim=True)
        return vectors.float().cpu().numpy().astype("float32")

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
        if image_path and not category:
            category = self._infer_visual_category(image_path)
        queries = self._query_vectors(image_path, description.strip(), category)
        macro_necklace = False
        layered_necklace = False
        visual_motif = ""
        if image_path and category.upper() == "KOLYE":
            with Image.open(image_path) as opened:
                width, height = ImageOps.exif_transpose(opened).size
            macro_necklace = max(width, height) / max(min(width, height), 1) <= 1.18
        grayscale_queries: np.ndarray | None = None
        if image_path:
            color_view_count = len(queries) // 2
            grayscale_queries = queries[color_view_count:]
            queries = queries[:color_view_count]
            if category.upper() == "KOLYE" and len(queries):
                necklace_type_scores = queries[0] @ self.necklace_type_vectors.T
                layered_necklace = bool(
                    necklace_type_scores[0] > necklace_type_scores[1]
                )
            if len(queries):
                motif_scores = (queries @ self.visual_motif_vectors.T).max(axis=0)
                order = np.argsort(motif_scores)[::-1]
                if (
                    motif_scores[order[0]] >= 0.28
                    and motif_scores[order[0]] - motif_scores[order[1]] >= 0.035
                ):
                    visual_motif = self.visual_motif_labels[int(order[0])]
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

        if (
            grayscale_queries is not None
            and self.object_index is not None
            and (category.upper() != "KOLYE" or macro_necklace)
        ):
            object_requested = min(
                self.object_index.ntotal,
                max(SEARCH_CANDIDATES, CATEGORY_SEARCH_CANDIDATES * 3 if category else 0),
            )
            object_scores, object_ids = self.object_index.search(
                grayscale_queries, object_requested
            )
            for view_scores, view_ids in zip(object_scores, object_ids):
                for item_id, score in zip(view_ids, view_scores):
                    if item_id < 0:
                        continue
                    weighted = float(score) * 1.04
                    fused[item_id] = max(fused.get(item_id, -1.0), weighted)

        dino_queries = (
            self._encode_dino_images(image_path, category)
            if image_path and self.dino_index is not None
            else None
        )
        if dino_queries is not None and category.upper() == "KOLYE":
            # Portrait shop photos may contain layered necklaces. The middle
            # overlapping crops preserve the number/order of chains and
            # pendants; tight crops are reserved for genuine macro photos.
            dino_queries = select_necklace_dino_queries(
                dino_queries,
                macro_necklace=macro_necklace,
                layered_necklace=layered_necklace,
            )
        if dino_queries is not None and self.dino_index is not None:
            dino_requested = min(
                self.dino_index.ntotal,
                max(SEARCH_CANDIDATES, CATEGORY_SEARCH_CANDIDATES * 3 if category else 0),
            )
            dino_scores, dino_ids = self.dino_index.search(
                dino_queries, dino_requested
            )
            for view_scores, view_ids in zip(dino_scores, dino_ids):
                for item_id, score in zip(view_ids, view_scores):
                    if item_id < 0:
                        continue
                    weighted = float(score) * 1.18
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
                if visual_motif:
                    metadata = " ".join(
                        [
                            str(product.get("aciklama", "")),
                            " ".join(
                                str(item) for item in product.get("arama_etiketleri", [])
                            ),
                        ]
                    ).casefold()
                    if visual_motif.casefold() in metadata:
                        adjusted += 0.10
            ranked.append((adjusted, product))

        ordered = sorted(ranked, key=lambda item: item[0], reverse=True)
        if image_path and visual_motif == "nazar gözü":
            with Image.open(image_path) as opened:
                query_motif_count = count_blue_motif_centers(
                    ImageOps.exif_transpose(opened)
                )
            if query_motif_count >= 3:
                geometry_ranked: list[tuple[float, dict[str, Any]]] = []
                for score, product in ordered[:100]:
                    reference_path = IMAGE_DIR / f"{product['kod']}.jpg"
                    reference_count = 0
                    if reference_path.is_file():
                        with Image.open(reference_path) as reference:
                            reference_count = count_blue_motif_centers(reference)
                    difference = abs(query_motif_count - reference_count)
                    bonus = 0.0
                    if reference_count >= 3:
                        bonus = max(0.0, 0.18 - (0.04 * difference))
                    geometry_ranked.append((score + bonus, product))
                ordered = sorted(
                    geometry_ranked + ordered[100:],
                    key=lambda item: item[0],
                    reverse=True,
                )

        results: list[dict[str, Any]] = []
        for score, product in ordered:
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

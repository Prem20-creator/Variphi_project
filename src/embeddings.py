"""
embeddings.py
-------------
Generates CLIP embeddings for video frames and text queries.
Uses openai/clip-vit-base-patch32 via the transformers library.
All embeddings are L2-normalised for cosine-similarity search.
"""

import logging
import time
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

# ── Model singleton ──────────────────────────────────────────────────────────
_model: CLIPModel = None
_processor: CLIPProcessor = None
_device: str = "cpu"


def _get_model() -> tuple[CLIPModel, CLIPProcessor, str]:
    """Lazy-load CLIP model (once per process)."""
    global _model, _processor, _device

    if _model is None:
        logger.info("Loading CLIP model (openai/clip-vit-base-patch32)…")
        t0 = time.time()

        model_id = "openai/clip-vit-base-patch32"
        _processor = CLIPProcessor.from_pretrained(model_id)
        _model = CLIPModel.from_pretrained(model_id)

        # Use MPS on Apple Silicon if available, else CPU
        if torch.backends.mps.is_available():
            _device = "mps"
        else:
            _device = "cpu"

        _model = _model.to(_device)
        _model.eval()

        logger.info(f"CLIP loaded in {time.time()-t0:.1f}s on {_device.upper()}")

    return _model, _processor, _device


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise a batch of row vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    return vectors / norms


# ── Public API ───────────────────────────────────────────────────────────────

def embed_frames(
    frame_paths: list[str],
    batch_size: int = 32,
) -> np.ndarray:
    """
    Generate CLIP image embeddings for a list of frame paths.

    Args:
        frame_paths:  Absolute paths to JPEG frame images.
        batch_size:   Number of images processed per forward pass.

    Returns:
        Float32 array of shape (N, 512), L2-normalised.
    """
    model, processor, device = _get_model()
    all_embeddings = []

    total = len(frame_paths)
    t0 = time.time()

    for i in range(0, total, batch_size):
        batch_paths = frame_paths[i : i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception as e:
                logger.warning(f"Skipping {p}: {e}")
                images.append(Image.new("RGB", (224, 224)))  # blank fallback

        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            feats = model.get_image_features(**inputs)

        batch_emb = feats.cpu().numpy().astype(np.float32)
        all_embeddings.append(batch_emb)

        done = min(i + batch_size, total)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        logger.info(f"  Embedded {done}/{total} frames | {rate:.1f} frames/sec")

    embeddings = np.vstack(all_embeddings)
    return _normalize(embeddings)


def embed_text(query: str) -> np.ndarray:
    """
    Generate a CLIP text embedding for a natural language query.

    Args:
        query:  Free-form text string.

    Returns:
        Float32 array of shape (1, 512), L2-normalised.
    """
    model, processor, device = _get_model()

    inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        feats = model.get_text_features(**inputs)

    emb = feats.cpu().numpy().astype(np.float32)
    return _normalize(emb)

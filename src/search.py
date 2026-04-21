"""
search.py
---------
Builds and queries a FAISS index over CLIP frame embeddings.
Uses IndexFlatIP (inner-product) on L2-normalised vectors,
which is mathematically equivalent to cosine similarity.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)

INDEX_FILE = "faiss.index"
META_FILE = "metadata.json"


# ── Index building ────────────────────────────────────────────────────────────

def build_index(
    embeddings: np.ndarray,
    metadata: list[dict],
    index_dir: str,
) -> None:
    """
    Build a FAISS flat inner-product index and persist it.

    Args:
        embeddings:  Float32 array (N, D), already L2-normalised.
        metadata:    List of frame metadata dicts (same order as embeddings).
        index_dir:   Directory where index + metadata are saved.
    """
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    dim = embeddings.shape[1]
    logger.info(f"Building FAISS IndexFlatIP | dim={dim} | vectors={len(embeddings)}")

    t0 = time.time()
    index = faiss.IndexFlatIP(dim)   # cosine sim via normalised inner product
    index.add(embeddings)

    faiss.write_index(index, str(index_dir / INDEX_FILE))

    with open(index_dir / META_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t0
    logger.info(
        f"Index built in {elapsed:.2f}s | "
        f"{index.ntotal} vectors stored at '{index_dir}'"
    )


def load_index(index_dir: str) -> tuple[faiss.Index, list[dict]]:
    """
    Load a previously built FAISS index and metadata from disk.

    Returns:
        (faiss_index, metadata_list)
    """
    index_dir = Path(index_dir)
    idx_path = index_dir / INDEX_FILE
    meta_path = index_dir / META_FILE

    if not idx_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"No index found at '{index_dir}'. "
            "Please upload and index a video first."
        )

    index = faiss.read_index(str(idx_path))
    with open(meta_path) as f:
        metadata = json.load(f)

    logger.info(f"Index loaded | {index.ntotal} vectors | {len(metadata)} frames")
    return index, metadata


# ── Querying ──────────────────────────────────────────────────────────────────

def search(
    query_embedding: np.ndarray,
    index: faiss.Index,
    metadata: list[dict],
    top_k: int = 10,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> list[dict]:
    """
    Search the FAISS index for the top-K most relevant frames.

    Args:
        query_embedding:  Float32 array (1, D), L2-normalised.
        index:            Loaded FAISS index.
        metadata:         Frame metadata list.
        top_k:            Number of results to return.
        start_sec:        Optional temporal filter — start of window (seconds).
        end_sec:          Optional temporal filter — end of window (seconds).

    Returns:
        List of result dicts sorted by score descending.
    """
    t0 = time.time()

    # Over-fetch when time-filter active so we can still return top_k after filtering
    fetch_k = top_k * 10 if (start_sec is not None or end_sec is not None) else top_k

    scores, indices = index.search(query_embedding, min(fetch_k, index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue

        frame_meta = metadata[idx]

        # Temporal filter
        ts = frame_meta["timestamp_sec"]
        if start_sec is not None and ts < start_sec:
            continue
        if end_sec is not None and ts > end_sec:
            continue

        results.append(
            {
                "frame_id": frame_meta["frame_id"],
                "frame_path": frame_meta["frame_path"],
                "timestamp_sec": ts,
                "timestamp_hms": frame_meta["timestamp_hms"],
                "score": float(round(score, 4)),
            }
        )

        if len(results) >= top_k:
            break

    latency_ms = (time.time() - t0) * 1000
    logger.info(
        f"Query returned {len(results)} results in {latency_ms:.1f} ms"
    )

    return results


# ── Results persistence ───────────────────────────────────────────────────────

def save_results(results: list[dict], query: str, output_path: str) -> None:
    """Append query results to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if output_path.exists():
        try:
            with open(output_path) as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = []

    entry = {"query": query, "results": results}
    existing.append(entry)

    with open(output_path, "w") as f:
        json.dump(existing, f, indent=2)

    logger.info(f"Results saved → {output_path}")

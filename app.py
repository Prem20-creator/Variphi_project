"""
app.py  —  Intelligent Video Search Engine
==========================================
Run with:  streamlit run app.py
"""

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.extract_frames import extract_frames
from src.embeddings import embed_frames, embed_text
from src.search import build_index, load_index, save_results, search

# ── Directories ─────────────────────────────────────────────────────────────
FRAMES_DIR = ROOT / "data" / "frames"
INDEX_DIR = ROOT / "index"
RESULTS_PATH = ROOT / "results" / "results.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VideoSearch · AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

    /* ── Reset & Base ── */
    html, body, [class*="css"] {
        font-family: 'DM Mono', monospace;
    }

    .stApp {
        background: #0a0a0f;
        color: #e8e8f0;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem; padding-bottom: 4rem; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0f0f1a !important;
        border-right: 1px solid #1e1e32;
    }
    [data-testid="stSidebar"] * { color: #c8c8e0 !important; }

    /* ── Hero header ── */
    .hero {
        text-align: center;
        padding: 3rem 1rem 2rem;
        position: relative;
    }
    .hero-eyebrow {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        color: #6e6ef0;
        margin-bottom: 0.75rem;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(2.4rem, 5vw, 4rem);
        font-weight: 800;
        line-height: 1.05;
        color: #ffffff;
        margin: 0 0 0.5rem;
    }
    .hero-title span {
        background: linear-gradient(135deg, #6e6ef0 0%, #a78bfa 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-sub {
        font-family: 'DM Mono', monospace;
        font-size: 0.82rem;
        color: #6b6b8a;
        letter-spacing: 0.05em;
    }

    /* ── Divider ── */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #2a2a4a, transparent);
        margin: 1.5rem 0;
    }

    /* ── Section labels ── */
    .section-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        color: #6e6ef0;
        margin-bottom: 0.5rem;
    }

    /* ── Status pill ── */
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-family: 'DM Mono', monospace;
        letter-spacing: 0.05em;
        font-weight: 500;
    }
    .pill-ready   { background: #0d2d1a; color: #4ade80; border: 1px solid #166534; }
    .pill-pending { background: #1a1a0d; color: #facc15; border: 1px solid #713f12; }
    .pill-dot { width:6px; height:6px; border-radius:50%; background: currentColor; }

    /* ── Result cards ── */
    .result-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
        gap: 1.25rem;
        margin-top: 1.5rem;
    }
    .result-card {
        background: #111120;
        border: 1px solid #1e1e35;
        border-radius: 12px;
        overflow: hidden;
        transition: border-color 0.2s, transform 0.2s;
        position: relative;
    }
    .result-card:hover {
        border-color: #6e6ef0;
        transform: translateY(-2px);
    }
    .result-card img {
        width: 100%;
        display: block;
        aspect-ratio: 16/9;
        object-fit: cover;
    }
    .card-body {
        padding: 0.75rem 1rem;
    }
    .card-ts {
        font-family: 'Syne', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #e8e8f0;
        margin-bottom: 0.25rem;
    }
    .card-score {
        font-size: 0.7rem;
        color: #6b6b8a;
        letter-spacing: 0.1em;
    }
    .score-bar-bg {
        height: 3px;
        background: #1e1e35;
        border-radius: 2px;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 2px;
        background: linear-gradient(90deg, #6e6ef0, #38bdf8);
    }
    .rank-badge {
        position: absolute;
        top: 0.5rem;
        left: 0.5rem;
        background: rgba(10,10,15,0.85);
        backdrop-filter: blur(4px);
        border: 1px solid #2a2a4a;
        border-radius: 6px;
        padding: 0.15rem 0.5rem;
        font-size: 0.65rem;
        font-family: 'DM Mono', monospace;
        color: #a78bfa;
        letter-spacing: 0.05em;
    }

    /* ── Metric boxes ── */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
        flex-wrap: wrap;
    }
    .metric-box {
        flex: 1;
        min-width: 120px;
        background: #111120;
        border: 1px solid #1e1e35;
        border-radius: 10px;
        padding: 0.85rem 1rem;
    }
    .metric-val {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #a78bfa;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.65rem;
        color: #6b6b8a;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }

    /* ── Streamlit overrides ── */
    .stButton > button {
        background: linear-gradient(135deg, #4f4fd0, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.08em !important;
        padding: 0.6rem 1.5rem !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }

    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stSlider > div {
        background: #111120 !important;
        border-color: #1e1e35 !important;
        color: #e8e8f0 !important;
        font-family: 'DM Mono', monospace !important;
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6e6ef0, #38bdf8) !important;
    }

    [data-testid="stFileUploader"] {
        background: #111120 !important;
        border: 1px dashed #2a2a4a !important;
        border-radius: 10px !important;
    }

    .stAlert {
        background: #111120 !important;
        border-color: #2a2a4a !important;
        border-radius: 10px !important;
    }

    /* ── Spinner text ── */
    .stSpinner > div { border-top-color: #6e6ef0 !important; }

    /* ── Checkbox ── */
    .stCheckbox label { font-size: 0.8rem !important; color: #9090b0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "indexed": False,
        "video_name": None,
        "n_frames": 0,
        "index_time": 0.0,
        "faiss_index": None,
        "frame_metadata": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Helper: clear old index ───────────────────────────────────────────────────
def _clear_index():
    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    for f in INDEX_DIR.glob("*"):
        f.unlink(missing_ok=True)

    st.session_state.indexed = False
    st.session_state.faiss_index = None
    st.session_state.frame_metadata = None
    st.session_state.video_name = None


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">▸ Variphi · AI Video Retrieval</div>
        <h1 class="hero-title">Search Video with<br><span>Natural Language</span></h1>
        <p class="hero-sub">CLIP · FAISS · Semantic Similarity · Sub-second Retrieval</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Upload + Settings
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<p style="font-family:\'Syne\',sans-serif;font-size:1.1rem;font-weight:700;'
        'color:#fff;margin-bottom:1rem;">⚙ Configuration</p>',
        unsafe_allow_html=True,
    )

    # ── Status pill ──
    if st.session_state.indexed:
        st.markdown(
            f'<div class="pill pill-ready"><span class="pill-dot"></span>'
            f'Indexed · {st.session_state.n_frames} frames</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="pill pill-pending"><span class="pill-dot"></span>'
            'No video indexed</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Upload ──
    st.markdown('<p class="section-label">📂 Video File</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">⚙ Index Settings</p>', unsafe_allow_html=True)

    sampling_fps = st.select_slider(
        "Frame sampling (frames / sec)",
        options=[0.25, 0.5, 1.0, 2.0],
        value=1.0,
    )
    st.caption(
        {
            0.25: "1 frame / 4 sec — fastest indexing",
            0.5:  "1 frame / 2 sec — balanced",
            1.0:  "1 frame / sec — recommended",
            2.0:  "2 frames / sec — most detail",
        }[sampling_fps]
    )

    batch_size = st.select_slider(
        "Embedding batch size",
        options=[8, 16, 32],
        value=16,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    index_btn = st.button("⚡  Index Video", use_container_width=True)

    if st.session_state.indexed:
        if st.button("🗑  Clear & Reset", use_container_width=True):
            _clear_index()
            st.rerun()

    # ── About ──
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.65rem;color:#3a3a5a;letter-spacing:0.08em;">'
        'CLIP · openai/clip-vit-base-patch32<br>'
        'FAISS IndexFlatIP · Cosine Similarity<br>'
        'OpenCV Frame Extraction<br>'
        'Streamlit UI</p>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# INDEXING
# ════════════════════════════════════════════════════════════════════════════
if index_btn:
    if uploaded_file is None:
        st.error("Please upload a video file first.")
    else:
        _clear_index()

        # Save uploaded video to disk
        video_path = ROOT / "data" / f"uploaded_{uploaded_file.name}"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="section-label">⚙ INDEXING · {uploaded_file.name}</p>',
            unsafe_allow_html=True,
        )

        t_total = time.time()

        # Step 1 — Extract frames
        with st.spinner("Extracting frames…"):
            progress = st.progress(0, text="Extracting frames from video…")
            metadata = extract_frames(
                video_path=str(video_path),
                output_dir=str(FRAMES_DIR),
                fps=sampling_fps,
            )
            progress.progress(33, text=f"Extracted {len(metadata)} frames ✓")

        # Step 2 — Embed frames
        with st.spinner("Generating CLIP embeddings…"):
            frame_paths = [m["frame_path"] for m in metadata]
            progress.progress(40, text="Loading CLIP model…")
            embeddings = embed_frames(frame_paths, batch_size=batch_size)
            progress.progress(85, text=f"Embedded {len(embeddings)} frames ✓")

        # Step 3 — Build FAISS index
        with st.spinner("Building FAISS index…"):
            build_index(embeddings, metadata, str(INDEX_DIR))
            progress.progress(100, text="Index ready ✓")

        elapsed = time.time() - t_total
        throughput = len(metadata) / elapsed

        # Load into session
        faiss_index, frame_meta = load_index(str(INDEX_DIR))
        st.session_state.indexed = True
        st.session_state.video_name = uploaded_file.name
        st.session_state.n_frames = len(metadata)
        st.session_state.index_time = elapsed
        st.session_state.faiss_index = faiss_index
        st.session_state.frame_metadata = frame_meta

        # Metrics
        st.markdown(
            f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-val">{len(metadata)}</div>
                    <div class="metric-label">Frames indexed</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{elapsed:.1f}s</div>
                    <div class="metric-label">Total index time</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{throughput:.1f}</div>
                    <div class="metric-label">Frames / sec</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{sampling_fps}</div>
                    <div class="metric-label">Sample FPS</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.success(f"✅ Video indexed successfully in {elapsed:.1f}s. Ready to search!")
        video_path.unlink(missing_ok=True)  # free disk space


# ════════════════════════════════════════════════════════════════════════════
# SEARCH INTERFACE
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">🔍 Natural Language Query</p>', unsafe_allow_html=True)

col_q, col_k = st.columns([4, 1])
with col_q:
    query = st.text_input(
        "Query",
        placeholder='e.g. "person near entrance carrying a bag"  ·  "red vehicle"  ·  "two people talking"',
        label_visibility="collapsed",
    )
with col_k:
    top_k = st.selectbox("Top K", [5, 10, 15, 20], index=1, label_visibility="collapsed")

# Time filter
use_time_filter = st.checkbox("Enable time-range filter")
start_sec, end_sec = None, None

if use_time_filter:
    tc1, tc2 = st.columns(2)
    with tc1:
        start_str = st.text_input("Start time (HH:MM:SS)", value="00:00:00")
    with tc2:
        end_str = st.text_input("End time (HH:MM:SS)", value="00:30:00")

    def _hms_to_sec(s: str) -> float:
        try:
            h, m, sec = s.strip().split(":")
            return int(h) * 3600 + int(m) * 60 + float(sec)
        except Exception:
            return 0.0

    start_sec = _hms_to_sec(start_str)
    end_sec = _hms_to_sec(end_str)

search_btn = st.button("🔍  Search", use_container_width=False)


# ── Run search ──────────────────────────────────────────────────────────────
if search_btn:
    if not st.session_state.indexed:
        st.error("No video indexed yet. Please upload and index a video in the sidebar.")
    elif not query.strip():
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Encoding query and searching…"):
            t0 = time.time()
            qemb = embed_text(query.strip())
            results = search(
                query_embedding=qemb,
                index=st.session_state.faiss_index,
                metadata=st.session_state.frame_metadata,
                top_k=top_k,
                start_sec=start_sec,
                end_sec=end_sec,
            )
            latency_ms = (time.time() - t0) * 1000

        # ── Query latency banner ──
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:1rem;margin:1rem 0 0.5rem;">
                <span style="font-family:'Syne',sans-serif;font-size:1.05rem;
                             font-weight:700;color:#fff;">
                    Results for &nbsp;<span style="color:#a78bfa;">"{query}"</span>
                </span>
                <span style="font-family:'DM Mono',monospace;font-size:0.7rem;
                             color:#6b6b8a;margin-left:auto;">
                    {latency_ms:.1f} ms &nbsp;·&nbsp; {len(results)} results
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not results:
            st.info("No results found. Try a different query or adjust the time filter.")
        else:
            # Save results
            RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            save_results(results, query, str(RESULTS_PATH))

            # ── Render result cards ──
            cols_per_row = 3
            for row_start in range(0, len(results), cols_per_row):
                row_results = results[row_start : row_start + cols_per_row]
                cols = st.columns(cols_per_row)

                for col, res in zip(cols, row_results):
                    rank = row_start + row_results.index(res) + 1
                    score_pct = min(100, int(res["score"] * 100))
                    score_color = (
                        "#4ade80" if res["score"] > 0.28
                        else "#facc15" if res["score"] > 0.22
                        else "#f87171"
                    )

                    with col:
                        frame_path = res["frame_path"]
                        if Path(frame_path).exists():
                            img = Image.open(frame_path)
                            st.image(img, use_container_width=True)
                        else:
                            st.markdown(
                                '<div style="background:#1a1a2e;aspect-ratio:16/9;'
                                'border-radius:8px;display:flex;align-items:center;'
                                'justify-content:center;color:#444;">'
                                'Frame unavailable</div>',
                                unsafe_allow_html=True,
                            )

                        st.markdown(
                            f"""
                            <div style="background:#111120;border:1px solid #1e1e35;
                                        border-radius:0 0 10px 10px;padding:0.75rem 1rem;
                                        margin-top:-6px;">
                                <div style="display:flex;align-items:center;
                                            justify-content:space-between;margin-bottom:0.4rem;">
                                    <span style="font-family:'Syne',sans-serif;
                                                 font-size:1.05rem;font-weight:700;
                                                 color:#fff;">
                                        {res["timestamp_hms"]}
                                    </span>
                                    <span style="font-family:'DM Mono',monospace;
                                                 font-size:0.65rem;color:#3a3a5a;
                                                 background:#0a0a0f;border:1px solid #1e1e35;
                                                 padding:0.1rem 0.4rem;border-radius:4px;">
                                        #{rank}
                                    </span>
                                </div>
                                <div style="font-size:0.68rem;color:#6b6b8a;
                                            letter-spacing:0.08em;margin-bottom:0.5rem;">
                                    SCORE &nbsp;
                                    <span style="color:{score_color};font-weight:600;">
                                        {res["score"]:.4f}
                                    </span>
                                </div>
                                <div style="height:3px;background:#1e1e35;
                                            border-radius:2px;overflow:hidden;">
                                    <div style="width:{score_pct}%;height:100%;
                                                background:linear-gradient(90deg,#6e6ef0,#38bdf8);
                                                border-radius:2px;"></div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # ── Results JSON link ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f'<p style="font-size:0.7rem;color:#3a3a5a;font-family:\'DM Mono\',monospace;">'
                f'Results saved → results/results.json</p>',
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════
if not st.session_state.indexed and not search_btn:
    st.markdown(
        """
        <div style="margin-top:3rem;padding:2rem;background:#0d0d1a;
                    border:1px solid #1a1a2e;border-radius:14px;">
            <p style="font-family:'Syne',sans-serif;font-size:0.95rem;
                      font-weight:600;color:#fff;margin-bottom:1rem;">
                How it works
            </p>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;">
                <div style="text-align:center;padding:1rem;">
                    <div style="font-size:1.8rem;margin-bottom:0.5rem;">📹</div>
                    <div style="font-family:'Syne',sans-serif;font-size:0.8rem;
                                font-weight:700;color:#a78bfa;margin-bottom:0.3rem;">
                        1. Upload
                    </div>
                    <div style="font-size:0.7rem;color:#6b6b8a;line-height:1.5;">
                        Upload any MP4, AVI, MOV or MKV video file
                    </div>
                </div>
                <div style="text-align:center;padding:1rem;">
                    <div style="font-size:1.8rem;margin-bottom:0.5rem;">🎞️</div>
                    <div style="font-family:'Syne',sans-serif;font-size:0.8rem;
                                font-weight:700;color:#a78bfa;margin-bottom:0.3rem;">
                        2. Index
                    </div>
                    <div style="font-size:0.7rem;color:#6b6b8a;line-height:1.5;">
                        Frames are extracted &amp; embedded with CLIP
                    </div>
                </div>
                <div style="text-align:center;padding:1rem;">
                    <div style="font-size:1.8rem;margin-bottom:0.5rem;">🔍</div>
                    <div style="font-family:'Syne',sans-serif;font-size:0.8rem;
                                font-weight:700;color:#a78bfa;margin-bottom:0.3rem;">
                        3. Query
                    </div>
                    <div style="font-size:0.7rem;color:#6b6b8a;line-height:1.5;">
                        Type any natural language query to search
                    </div>
                </div>
                <div style="text-align:center;padding:1rem;">
                    <div style="font-size:1.8rem;margin-bottom:0.5rem;">⚡</div>
                    <div style="font-family:'Syne',sans-serif;font-size:0.8rem;
                                font-weight:700;color:#a78bfa;margin-bottom:0.3rem;">
                        4. Retrieve
                    </div>
                    <div style="font-size:0.7rem;color:#6b6b8a;line-height:1.5;">
                        Top-K frames returned with timestamps &amp; scores
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

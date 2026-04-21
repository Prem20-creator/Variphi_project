"""
extract_frames.py
-----------------
Extracts frames from a video file at a configurable rate.
Stores frames as JPEG images with metadata for later retrieval.
"""

import cv2
import os
import json
import logging
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def seconds_to_hms(seconds: float) -> str:
    """Convert seconds to HH:MM:SS string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = 1.0,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> list[dict]:
    """
    Extract frames from a video file.

    Args:
        video_path:  Path to the input video file.
        output_dir:  Directory where extracted frames are saved.
        fps:         Frames to extract per second of video (default 1).
        start_time:  Optional start offset in seconds.
        end_time:    Optional end offset in seconds.

    Returns:
        List of metadata dicts: {frame_id, frame_path, timestamp_sec, timestamp_hms}
    """
    video_path = str(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / native_fps if native_fps > 0 else 0

    logger.info(f"Video: {os.path.basename(video_path)}")
    logger.info(f"Duration: {seconds_to_hms(duration)} | Native FPS: {native_fps:.2f}")
    logger.info(f"Extracting at {fps} frame(s)/sec")

    # Frame interval in terms of video frames
    frame_interval = max(1, int(native_fps / fps))

    start_frame = int((start_time or 0) * native_fps)
    end_frame = int((end_time or duration) * native_fps)
    end_frame = min(end_frame, total_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    metadata = []
    frame_count = 0
    extracted = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_idx = start_frame + frame_count

        if current_frame_idx > end_frame:
            break

        if frame_count % frame_interval == 0:
            timestamp_sec = current_frame_idx / native_fps
            frame_id = f"frame_{extracted:06d}"
            frame_filename = f"{frame_id}.jpg"
            frame_path = str(output_dir / frame_filename)

            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            metadata.append(
                {
                    "frame_id": frame_id,
                    "frame_path": frame_path,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "timestamp_hms": seconds_to_hms(timestamp_sec),
                }
            )
            extracted += 1

            if extracted % 50 == 0:
                elapsed = time.time() - t0
                rate = extracted / elapsed
                logger.info(f"  Extracted {extracted} frames | {rate:.1f} frames/sec")

        frame_count += 1

    cap.release()

    elapsed = time.time() - t0
    logger.info(
        f"Done. {extracted} frames extracted in {elapsed:.1f}s "
        f"({extracted/elapsed:.1f} frames/sec)"
    )

    # Persist metadata alongside frames
    meta_path = output_dir / "frame_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata

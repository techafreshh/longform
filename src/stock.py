"""Stock footage module — Pexels API for optional B-roll footage."""

import os
import requests
from pathlib import Path
from typing import Optional

from .config import PEXELS_API_KEY


PEXELS_VIDEO_SEARCH_URL = "https://api.pexels.com/videos/search"


def search_stock_footage(
    query: str,
    output_dir: Path,
    count: int = 1,
    orientation: str = "landscape",
    min_duration: int = 5,
    max_duration: int = 30,
    verbose: bool = True,
) -> list[Path]:
    """
    Search and download stock video clips from Pexels.

    Args:
        query: Search terms for the footage.
        output_dir: Where to save downloaded clips.
        count: Number of clips to download.
        orientation: "landscape", "portrait", or "square".
        min_duration: Minimum clip duration in seconds.
        max_duration: Maximum clip duration in seconds.
        verbose: Print progress.

    Returns:
        List of paths to downloaded video files.
    """
    if not PEXELS_API_KEY:
        if verbose:
            print("  ℹ️ PEXELS_API_KEY not set — skipping stock footage")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    headers = {"Authorization": PEXELS_API_KEY}
    params = {
        "query": query,
        "per_page": count * 2,  # Fetch extra for filtering
        "orientation": orientation,
    }

    if verbose:
        print(f"  🎥 Searching Pexels for: '{query}'...")

    try:
        response = requests.get(PEXELS_VIDEO_SEARCH_URL, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        if verbose:
            print(f"  ⚠️ Pexels search failed: {e}")
        return []

    videos = data.get("videos", [])
    if not videos:
        if verbose:
            print(f"  ℹ️ No results found for '{query}'")
        return []

    # Filter by duration
    filtered = [
        v for v in videos
        if min_duration <= v.get("duration", 0) <= max_duration
    ]

    if not filtered:
        filtered = videos[:count]  # Use unfiltered if nothing matches

    downloaded = []
    for i, video in enumerate(filtered[:count]):
        # Find the HD video file
        video_files = video.get("video_files", [])
        # Prefer HD quality
        selected = None
        for vf in video_files:
            if vf.get("quality") == "hd":
                selected = vf
                break
        if not selected and video_files:
            selected = video_files[0]

        if not selected:
            continue

        download_url = selected.get("link")
        if not download_url:
            continue

        file_ext = download_url.split(".")[-1].split("?")[0] or "mp4"
        filename = output_dir / f"broll_{i + 1:02d}.{file_ext}"

        if verbose:
            print(f"  ⬇️ Downloading B-roll clip {i + 1}...")

        try:
            r = requests.get(download_url, stream=True)
            r.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            downloaded.append(filename)
        except Exception as e:
            if verbose:
                print(f"  ⚠️ Download failed: {e}")

    if verbose:
        print(f"  ✅ Downloaded {len(downloaded)} B-roll clip(s)")

    return downloaded

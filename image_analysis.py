"""Image processing utilities for advertisement analysis.

This module downloads ad images and performs:
- grayscale conversion
- Canny edge detection
- color distribution analysis
- dominant color detection
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests


def _download_image(image_url: str, timeout: int = 15) -> Optional[np.ndarray]:
    """Download an image URL and decode it as an OpenCV BGR image array."""
    try:
        response = requests.get(image_url, timeout=timeout)
        response.raise_for_status()
        image_data = np.frombuffer(response.content, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        return image_bgr
    except Exception:
        return None


def _dominant_colors(image_rgb: np.ndarray, k: int = 3) -> List[Tuple[Tuple[int, int, int], float]]:
    """Find dominant RGB colors using k-means clustering.

    Returns:
        List of ((R, G, B), percentage) tuples sorted by frequency.
    """
    pixels = image_rgb.reshape(-1, 3).astype(np.float32)
    if len(pixels) < k:
        k = max(1, len(pixels))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS,
    )

    counts = Counter(labels.flatten().tolist())
    total = float(len(labels))

    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    results: List[Tuple[Tuple[int, int, int], float]] = []
    for idx, count in ranked:
        color = tuple(int(v) for v in centers[idx])
        percentage = (count / total) * 100
        results.append((color, percentage))

    return results


def _channel_histograms(image_rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute normalized histogram for each RGB color channel."""
    histograms: Dict[str, np.ndarray] = {}
    channel_map = {"R": 0, "G": 1, "B": 2}

    for channel_name, idx in channel_map.items():
        hist = cv2.calcHist([image_rgb], [idx], None, [256], [0, 256]).flatten()
        hist = hist / max(hist.sum(), 1.0)
        histograms[channel_name] = hist

    return histograms


def analyze_image(image_url: str) -> Optional[Dict]:
    """Analyze a single advertisement image URL."""
    image_bgr = _download_image(image_url)
    if image_bgr is None:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    histograms = _channel_histograms(image_rgb)
    dominant = _dominant_colors(image_rgb, k=3)

    h, w = image_rgb.shape[:2]

    return {
        "image_url": image_url,
        "original_rgb": image_rgb,
        "gray": gray,
        "edges": edges,
        "histograms": histograms,
        "dominant_colors": dominant,
        "width": int(w),
        "height": int(h),
        "pixel_count": int(w * h),
    }


def analyze_images(image_urls: List[str]) -> List[Dict]:
    """Analyze a list of image URLs and return successful analyses only."""
    analyses: List[Dict] = []
    for url in image_urls:
        result = analyze_image(url)
        if result is not None:
            analyses.append(result)
    return analyses

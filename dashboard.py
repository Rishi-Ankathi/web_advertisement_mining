"""Dashboard helper functions for charts and aggregated statistics."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def ad_type_distribution_chart(ads_df: pd.DataFrame) -> go.Figure:
    """Create a pie chart for advertisement type distribution."""
    if ads_df.empty:
        return go.Figure()

    counts = ads_df["ad_type"].value_counts().reset_index()
    counts.columns = ["ad_type", "count"]

    fig = px.pie(
        counts,
        values="count",
        names="ad_type",
        title="Advertisement Type Distribution",
        hole=0.35,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def color_usage_chart(image_analyses: List[Dict]) -> go.Figure:
    """Create a bar chart for dominant color usage across analyzed ads."""
    if not image_analyses:
        return go.Figure()

    color_counter: Counter[str] = Counter()

    for analysis in image_analyses:
        for (r, g, b), pct in analysis.get("dominant_colors", []):
            # Quantize colors so similar shades are grouped together.
            qr, qg, qb = (int(v // 32 * 32) for v in (r, g, b))
            hex_color = f"#{qr:02x}{qg:02x}{qb:02x}"
            color_counter[hex_color] += float(pct)

    if not color_counter:
        return go.Figure()

    top_colors = color_counter.most_common(10)
    df = pd.DataFrame(top_colors, columns=["color", "usage_score"])

    fig = px.bar(
        df,
        x="color",
        y="usage_score",
        title="Color Usage in Advertisements (Top Dominant Colors)",
        color="color",
        color_discrete_map={c: c for c in df["color"]},
    )
    fig.update_layout(showlegend=False, xaxis_title="Color", yaxis_title="Usage Score")
    return fig


def advertisement_frequency_chart(ads_df: pd.DataFrame) -> go.Figure:
    """Create a frequency chart for advertisements by record index/time order."""
    if ads_df.empty:
        return go.Figure()

    freq_df = ads_df.copy().reset_index(drop=True)
    freq_df["ad_number"] = np.arange(1, len(freq_df) + 1)

    fig = px.histogram(
        freq_df,
        x="ad_type",
        color="ad_type",
        title="Advertisement Frequency Chart",
    )
    fig.update_layout(xaxis_title="Ad Type", yaxis_title="Frequency", showlegend=False)
    return fig


def histogram_figure(channel_histograms: Dict[str, np.ndarray], title: str) -> go.Figure:
    """Create RGB histogram line chart for one image."""
    fig = go.Figure()
    x = list(range(256))
    color_map = {"R": "red", "G": "green", "B": "blue"}

    for channel_name in ["R", "G", "B"]:
        values = channel_histograms.get(channel_name)
        if values is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=x,
                y=values,
                mode="lines",
                name=f"{channel_name} Channel",
                line=dict(color=color_map[channel_name]),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Pixel Intensity (0-255)",
        yaxis_title="Normalized Frequency",
    )
    return fig


def compute_statistics(ads_df: pd.DataFrame, image_analyses: List[Dict]) -> Tuple[int, List[str], str]:
    """Compute summary statistics for the dashboard metrics section."""
    ad_count = int(len(ads_df))

    color_counter: Counter[str] = Counter()
    widths: List[int] = []
    heights: List[int] = []

    for analysis in image_analyses:
        widths.append(int(analysis.get("width", 0)))
        heights.append(int(analysis.get("height", 0)))
        for (r, g, b), pct in analysis.get("dominant_colors", []):
            hex_color = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
            color_counter[hex_color] += float(pct)

    most_common_colors = [c for c, _ in color_counter.most_common(3)]

    if widths and heights:
        avg_w = int(np.mean(widths))
        avg_h = int(np.mean(heights))
        avg_size = f"{avg_w} x {avg_h} px"
    else:
        avg_size = "N/A"

    return ad_count, most_common_colors, avg_size

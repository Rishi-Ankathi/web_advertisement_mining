"""Streamlit application: Web Advertisement Mining and Analysis."""

from __future__ import annotations

from typing import List

import pandas as pd
import streamlit as st

from dashboard import (
    ad_type_distribution_chart,
    advertisement_frequency_chart,
    color_usage_chart,
    compute_statistics,
    histogram_figure,
)
from image_analysis import analyze_images
from web_scraper import extract_advertisements


st.set_page_config(page_title="Web Advertisement Mining and Analysis", layout="wide")


def _collect_unique_image_urls(ads_df: pd.DataFrame) -> List[str]:
    if ads_df.empty or "image_url" not in ads_df.columns:
        return []
    urls = ads_df["image_url"].dropna().astype(str).unique().tolist()
    return [u for u in urls if u.strip()]


def main() -> None:
    st.title("Web Advertisement Mining and Analysis")
    st.markdown(
        "Analyze advertisement text, links, banners, and images from any webpage URL."
    )

    with st.sidebar:
        st.header("Input")
        url = st.text_input("Enter Website URL", placeholder="https://example.com")
        run_btn = st.button("Mine Advertisements", type="primary")

    if not run_btn:
        st.info("Enter a URL and click **Mine Advertisements** to start analysis.")
        return

    if not url.strip():
        st.warning("Please provide a valid website URL.")
        return

    with st.spinner("Scraping advertisements from webpage..."):
        try:
            ads_df = extract_advertisements(url.strip())
        except Exception as exc:
            st.error(f"Failed to scrape webpage: {exc}")
            return

    if ads_df.empty:
        st.warning("No advertisement-like content was detected on this page.")
        return

    image_urls = _collect_unique_image_urls(ads_df)
    with st.spinner("Analyzing advertisement images..."):
        image_analyses = analyze_images(image_urls)

    # ------------------------------
    # Section 1 – Extracted Advertisements
    # ------------------------------
    st.header("Section 1 – Extracted Advertisements")
    st.dataframe(ads_df, use_container_width=True)

    for idx, row in ads_df.iterrows():
        with st.expander(f"Ad #{idx + 1} - {row.get('ad_type', 'unknown')}"):
            col1, col2 = st.columns([1, 2])
            with col1:
                if pd.notna(row.get("image_url")):
                    st.image(row["image_url"], caption="Advertisement Image", use_container_width=True)
            with col2:
                st.write("**Advertisement Text:**")
                st.write(row.get("text") if pd.notna(row.get("text")) else "N/A")
                st.write("**Link:**")
                link = row.get("link_url")
                if pd.notna(link):
                    st.markdown(f"[{link}]({link})")
                else:
                    st.write("N/A")

    # ------------------------------
    # Section 2 – Advertisement Image Analysis
    # ------------------------------
    st.header("Section 2 – Advertisement Image Analysis")
    if not image_analyses:
        st.info("No valid advertisement images available for processing.")
    else:
        for i, analysis in enumerate(image_analyses, start=1):
            st.subheader(f"Image Analysis #{i}")
            c1, c2 = st.columns(2)
            with c1:
                st.image(
                    analysis["edges"],
                    caption="Edge Detection (Canny)",
                    use_container_width=True,
                    clamp=True,
                )
            with c2:
                hist_fig = histogram_figure(
                    analysis["histograms"],
                    title="Color Distribution (RGB Histogram)",
                )
                st.plotly_chart(hist_fig, use_container_width=True)

            dom_colors = analysis.get("dominant_colors", [])
            if dom_colors:
                chips = "  ".join(
                    [
                        f"`#{r:02x}{g:02x}{b:02x}` ({pct:.1f}%)"
                        for (r, g, b), pct in dom_colors
                    ]
                )
                st.write(f"**Dominant Colors:** {chips}")

    # ------------------------------
    # Section 3 – Advertisement Statistics
    # ------------------------------
    st.header("Section 3 – Advertisement Statistics")
    ad_count, common_colors, avg_img_size = compute_statistics(ads_df, image_analyses)
    m1, m2, m3 = st.columns(3)
    m1.metric("Number of Ads Detected", ad_count)
    m2.metric("Most Common Colors", ", ".join(common_colors) if common_colors else "N/A")
    m3.metric("Average Image Size", avg_img_size)

    # ------------------------------
    # Section 4 – Charts
    # ------------------------------
    st.header("Section 4 – Charts")
    fig1 = ad_type_distribution_chart(ads_df)
    fig2 = color_usage_chart(image_analyses)
    fig3 = advertisement_frequency_chart(ads_df)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        st.plotly_chart(fig2, use_container_width=True)

    st.plotly_chart(fig3, use_container_width=True)


if __name__ == "__main__":
    main()

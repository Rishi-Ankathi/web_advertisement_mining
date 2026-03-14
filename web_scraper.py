"""Web scraping utilities for advertisement mining.

This module is responsible for:
1. Downloading webpage HTML content
2. Identifying likely advertisement elements
3. Returning extracted data in a structured pandas DataFrame
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag


AD_KEYWORDS = (
    "ad",
    "ads",
    "advert",
    "sponsor",
    "sponsored",
    "promo",
    "promotion",
    "banner",
    "offer",
    "deal",
)


@dataclass
class AdRecord:
    """A normalized ad data record."""

    ad_type: str
    text: str
    image_url: Optional[str]
    link_url: Optional[str]
    source_url: str


def _fetch_html(url: str, timeout: int = 15) -> str:
    """Fetch raw HTML with a browser-like user agent."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text


def _tag_text(tag: Tag) -> str:
    """Safely extract condensed text from an element."""
    return " ".join(tag.get_text(" ", strip=True).split())


def _has_ad_indicator(tag: Tag) -> bool:
    """Check class/id/attributes for advertisement hints."""
    raw_parts: List[str] = []

    tag_id = tag.get("id")
    if tag_id:
        raw_parts.append(str(tag_id))

    classes = tag.get("class", [])
    if classes:
        raw_parts.extend(map(str, classes))

    aria_label = tag.get("aria-label")
    if aria_label:
        raw_parts.append(str(aria_label))

    combined = " ".join(raw_parts).lower()
    return any(keyword in combined for keyword in AD_KEYWORDS)


def _extract_image_ads(soup: BeautifulSoup, base_url: str) -> List[AdRecord]:
    """Extract ad-like image tags and linked images."""
    records: List[AdRecord] = []

    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if not src:
            continue

        image_url = urljoin(base_url, src)
        alt_text = (img.get("alt") or "").strip()
        parent_link = img.find_parent("a")
        link_url = urljoin(base_url, parent_link["href"]) if parent_link and parent_link.get("href") else None

        # Heuristics for ad detection.
        ad_like = _has_ad_indicator(img)
        if parent_link and _has_ad_indicator(parent_link):
            ad_like = True
        if alt_text and any(k in alt_text.lower() for k in AD_KEYWORDS):
            ad_like = True

        # Also treat very wide images as likely banner advertisements.
        width = str(img.get("width", "")).strip()
        height = str(img.get("height", "")).strip()
        if width.isdigit() and height.isdigit() and int(width) >= 468 and int(height) <= 120:
            ad_like = True

        if ad_like:
            records.append(
                AdRecord(
                    ad_type="image/banner",
                    text=alt_text,
                    image_url=image_url,
                    link_url=link_url,
                    source_url=base_url,
                )
            )

    return records


def _extract_promotional_sections(soup: BeautifulSoup, base_url: str) -> List[AdRecord]:
    """Extract sections/divs/articles likely representing promotions."""
    records: List[AdRecord] = []

    for tag in soup.find_all(["section", "div", "article", "aside"]):
        if not _has_ad_indicator(tag):
            continue

        text = _tag_text(tag)
        if len(text) < 20:
            continue

        img = tag.find("img")
        image_url = None
        if img:
            src = img.get("src") or img.get("data-src")
            if src:
                image_url = urljoin(base_url, src)

        link = tag.find("a", href=True)
        link_url = urljoin(base_url, link["href"]) if link else None

        records.append(
            AdRecord(
                ad_type="promotional_section",
                text=text,
                image_url=image_url,
                link_url=link_url,
                source_url=base_url,
            )
        )

    return records


def _extract_sponsored_links(soup: BeautifulSoup, base_url: str) -> List[AdRecord]:
    """Extract links that look like sponsored/promotional content."""
    records: List[AdRecord] = []

    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "")
        text = _tag_text(anchor)
        lower = f"{href} {text}".lower()

        if not any(k in lower for k in ("sponsor", "promo", "advert", "utm_", "campaign", "ref=")):
            continue

        records.append(
            AdRecord(
                ad_type="sponsored_link",
                text=text,
                image_url=None,
                link_url=urljoin(base_url, href),
                source_url=base_url,
            )
        )

    return records


def _deduplicate_records(records: List[AdRecord]) -> List[AdRecord]:
    """Remove duplicate ad records based on key fields."""
    seen = set()
    unique: List[AdRecord] = []
    for rec in records:
        key = (rec.ad_type, rec.text, rec.image_url, rec.link_url)
        if key in seen:
            continue
        seen.add(key)
        unique.append(rec)
    return unique


def extract_advertisements(url: str) -> pd.DataFrame:
    """Main extraction API used by Streamlit app.

    Args:
        url: Target webpage URL.

    Returns:
        pandas.DataFrame with columns:
            ad_type, text, image_url, link_url, source_url
    """
    parsed = urlparse(url)
    if not parsed.scheme:
        url = f"https://{url}"

    html = _fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    records: List[AdRecord] = []
    records.extend(_extract_image_ads(soup, url))
    records.extend(_extract_promotional_sections(soup, url))
    records.extend(_extract_sponsored_links(soup, url))

    clean_records = _deduplicate_records(records)
    df = pd.DataFrame([r.__dict__ for r in clean_records])

    if df.empty:
        return pd.DataFrame(columns=["ad_type", "text", "image_url", "link_url", "source_url"])

    # Normalize empty strings to None for consistent handling.
    for col in ["text", "image_url", "link_url"]:
        df[col] = df[col].replace("", None)
    return df

"""Scraper for Trustpilot reviews of Alan (https://fr.trustpilot.com/review/alan.com).

Uses two complementary data sources per page:
- **JSON-LD** structured data (``@graph`` items of type ``Review``) for the
  20 regular paginated reviews – these contain full untruncated text + title.
- **HTML** card parsing for the *carousel* (featured) reviews that appear at
  the top of the page – these are *not* included in the JSON-LD data.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup, NavigableString

from .base import Review, COMMON_HEADERS, polite_sleep, filter_reviews_by_date

logger = logging.getLogger(__name__)

TRUSTPILOT_BASE_URL = "https://fr.trustpilot.com/review/alan.com"


# ---------------------------------------------------------------------------
# JSON-LD extraction (primary source for regular / paginated reviews)
# ---------------------------------------------------------------------------

def _extract_reviews_from_json_ld(soup: BeautifulSoup) -> List[Review]:
    """Extract reviews from the ``@graph`` JSON-LD blocks embedded in the page.

    Trustpilot embeds the 20 paginated reviews as separate ``Review`` items
    inside a ``@graph`` array.  These contain the **full, untruncated** review
    body, headline, author, date and rating – much more reliable than HTML
    scraping.
    """
    reviews: List[Review] = []

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue

        graph: list = []
        if isinstance(data, dict) and "@graph" in data:
            graph = data["@graph"]
        elif isinstance(data, list):
            graph = data

        for item in graph:
            if not isinstance(item, dict):
                continue
            if item.get("@type") != "Review":
                continue

            try:
                # Date
                raw_date = item.get("datePublished", "")
                review_date = ""
                if raw_date:
                    dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                    review_date = dt.strftime("%Y-%m-%d")

                # Rating
                rating_val = float(
                    item.get("reviewRating", {}).get("ratingValue", 0)
                )

                # Author
                author_obj = item.get("author", {})
                author_name = (
                    author_obj.get("name", "Unknown")
                    if isinstance(author_obj, dict)
                    else str(author_obj)
                )

                headline = item.get("headline", "")
                body = item.get("reviewBody", "")

                reviews.append(
                    Review(
                        source="trustpilot",
                        author=author_name,
                        rating=rating_val,
                        date=review_date,
                        title=headline,
                        review_text=body,
                        language="fr",
                    )
                )
            except Exception as exc:
                logger.debug("Trustpilot JSON-LD: skipped item: %s", exc)
                continue

    return reviews


# ---------------------------------------------------------------------------
# HTML parsing helpers (for carousel cards only)
# ---------------------------------------------------------------------------

def _is_carousel_card(card) -> bool:
    """Return True if the card is a carousel (featured) card."""
    classes = card.get("class", [])
    # Match the carousel-specific class (hash suffix may change over time)
    return any("carouselReviewCard" in cls for cls in classes)


def _parse_carousel_card(card) -> Optional[Review]:
    """Parse a carousel (featured) review card from the HTML."""
    try:
        # --- Author ---
        author_el = card.select_one("[data-consumer-name-typography]")
        author = author_el.get_text(strip=True) if author_el else "Unknown"

        # --- Rating (from img alt: "Noté 5 sur 5 étoiles") ---
        rating = 0.0
        img = card.select_one("img[alt*='oté'], img[alt*='ated']")
        if img:
            for word in img.get("alt", "").split():
                try:
                    val = float(word)
                    if 1 <= val <= 5:
                        rating = val
                        break
                except ValueError:
                    continue

        # --- Date ---
        time_el = (
            card.select_one("time[data-service-review-date-time-ago]")
            or card.select_one("time")
        )
        review_date_str = ""
        if time_el and time_el.get("datetime"):
            dt = datetime.fromisoformat(
                time_el["datetime"].replace("Z", "+00:00")
            )
            review_date_str = dt.strftime("%Y-%m-%d")

        # --- Review text (skip "Voir plus" span) ---
        review_text = ""
        text_el = card.select_one("[data-relevant-review-text-typography]")
        if text_el:
            parts = [
                child.strip()
                for child in text_el.children
                if isinstance(child, NavigableString) and child.strip()
            ]
            review_text = "\n".join(parts)

        # Carousel cards have no separate title element.
        return Review(
            source="trustpilot",
            author=author,
            rating=rating,
            date=review_date_str,
            title="",
            review_text=review_text,
            language="fr",
        )
    except Exception as exc:
        logger.warning("Trustpilot: failed to parse carousel card: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_trustpilot(
    start_date: date,
    end_date: date,
    max_pages: int = 50,
    progress_callback=None,
) -> List[Review]:
    """Scrape Trustpilot reviews for Alan within a date range.

    Strategy per page:
    1. Parse **JSON-LD** ``@graph`` for the 20 regular paginated reviews
       (full text, title, exact date, rating).
    2. Parse **HTML** carousel cards for the featured reviews that only
       appear visually at the top (not in JSON-LD).  These are collected
       on page 1 only (they repeat on every page).

    Pagination stops when the oldest review on a page falls before
    *start_date* or *max_pages* is reached.
    """
    all_reviews: List[Review] = []
    seen_keys: set[str] = set()  # (author, date) for dedup
    session = requests.Session()
    session.headers.update(COMMON_HEADERS)

    def _add_unique(reviews: List[Review]) -> int:
        added = 0
        for r in reviews:
            key = (r.author.strip().lower(), r.date)
            if key not in seen_keys:
                seen_keys.add(key)
                all_reviews.append(r)
                added += 1
        return added

    for page_num in range(1, max_pages + 1):
        url = f"{TRUSTPILOT_BASE_URL}?page={page_num}"
        logger.info("Trustpilot: fetching page %d – %s", page_num, url)

        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Trustpilot: request failed for page %d: %s", page_num, exc)
            break

        soup = BeautifulSoup(resp.text, "lxml")

        # 1) JSON-LD: primary data for the 20 regular reviews
        jsonld_reviews = _extract_reviews_from_json_ld(soup)
        added_jsonld = _add_unique(jsonld_reviews)

        # 2) HTML carousel cards (page 1 only – they repeat on every page)
        added_carousel = 0
        if page_num == 1:
            all_cards = soup.select("article[data-service-review-card-paper]")
            for card in all_cards:
                if _is_carousel_card(card):
                    review = _parse_carousel_card(card)
                    if review:
                        added_carousel += _add_unique([review])

        total_page = added_jsonld + added_carousel
        if total_page == 0 and page_num > 1:
            logger.info("Trustpilot: no new reviews on page %d, stopping.", page_num)
            break

        if progress_callback:
            progress_callback(
                f"Trustpilot: page {page_num} → {added_jsonld} regular + "
                f"{added_carousel} carousel reviews"
            )

        # Early stop: check if oldest review on this page is before start_date
        page_dates = []
        for r in jsonld_reviews:
            if r.date:
                try:
                    page_dates.append(datetime.strptime(r.date, "%Y-%m-%d").date())
                except ValueError:
                    pass
        if page_dates and min(page_dates) < start_date:
            logger.info(
                "Trustpilot: oldest review on page %d (%s) < start_date (%s), stopping.",
                page_num,
                min(page_dates),
                start_date,
            )
            break

        polite_sleep(1.5, 3.0)

    # Final date-range filter
    filtered = filter_reviews_by_date(all_reviews, start_date, end_date)
    logger.info("Trustpilot: %d reviews after date filtering.", len(filtered))
    return filtered

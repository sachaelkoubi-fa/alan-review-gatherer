"""Scraper for Opinion Assurances reviews of Alan.

URL : https://www.opinion-assurances.fr/assureur-alan.html
Pagination: assureur-alan-page2.html … assureur-alan-page{N}.html

The site requires a custom TLS adapter (``DEFAULT@SECLEVEL=1``) and a
``Referer`` header for pagination to work.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context
from bs4 import BeautifulSoup

from .base import Review, COMMON_HEADERS, polite_sleep, filter_reviews_by_date

logger = logging.getLogger(__name__)

OPINION_ASSURANCES_BASE_URL = "https://www.opinion-assurances.fr/assureur-alan.html"

# French month names for date parsing
FRENCH_MONTHS = {
    "janvier": 1, "février": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "août": 8,
    "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12,
    "janv": 1, "févr": 2, "avr": 4, "juil": 7,
    "sept": 9, "oct": 10, "nov": 11, "déc": 12,
}


# ---------------------------------------------------------------------------
# TLS adapter – Opinion Assurances requires a lower security level
# ---------------------------------------------------------------------------

class _TLSAdapter(HTTPAdapter):
    """HTTPS adapter that lowers the TLS security level for compatibility."""

    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.set_ciphers("DEFAULT@SECLEVEL=1")
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date_from_description(text: str) -> Optional[str]:
    """Extract a date from the description line.

    Typical format:
        "Avis publié le 13/02/2026 suite à une expérience le 13/02/2026"
    """
    # DD/MM/YYYY
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if m:
        day, month, year = m.groups()
        try:
            d = date(int(year), int(month), int(day))
            return d.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # ISO YYYY-MM-DD
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        return m.group(0)

    # "DD month_name YYYY"
    text_lower = text.lower()
    for month_name, month_num in FRENCH_MONTHS.items():
        pattern = rf"(\d{{1,2}})\s+{re.escape(month_name)}\.?\s+(\d{{4}})"
        m = re.search(pattern, text_lower)
        if m:
            day, year = m.groups()
            try:
                d = date(int(year), month_num, int(day))
                return d.strftime("%Y-%m-%d")
            except ValueError:
                pass

    return None


def _extract_reviews_from_page(soup: BeautifulSoup) -> List[Review]:
    """Extract reviews from an Opinion Assurances page.

    Each review lives inside a ``<div itemprop="review">`` container with:
    - ``meta[itemprop="ratingValue"]`` → overall rating (1-5)
    - ``[itemprop="author"] [itemprop="name"]`` → author name
    - ``.oa_description`` → date string
    - ``h4.oa_text`` → review body
    - Stars: count ``i.fas.fa-star.active`` in ``.oa_stackLevel``
    """
    reviews: List[Review] = []

    containers = soup.select('[itemprop="review"]')
    if not containers:
        return reviews

    for container in containers:
        try:
            # --- Author ---
            author_el = container.select_one('[itemprop="name"]')
            author = author_el.get_text(strip=True) if author_el else "Anonyme"

            # --- Rating ---
            rating = 0.0
            rating_meta = container.select_one('meta[itemprop="ratingValue"]')
            if rating_meta and rating_meta.get("content"):
                try:
                    rating = float(rating_meta["content"])
                except ValueError:
                    pass
            # Fallback: count active stars in the first oa_stackLevel
            if rating == 0.0:
                stack = container.select_one(".oa_stackLevel")
                if stack:
                    active = stack.select("i.fas.fa-star.active")
                    if active:
                        rating = float(len(active))

            # --- Date ---
            review_date = ""
            desc_el = container.select_one(".oa_description")
            if desc_el:
                parsed = _parse_date_from_description(desc_el.get_text())
                if parsed:
                    review_date = parsed

            # --- Review text (in h4.oa_text) ---
            text_el = container.select_one("h4.oa_text")
            review_text = text_el.get_text(strip=True) if text_el else ""

            if review_text:
                reviews.append(
                    Review(
                        source="opinion_assurances",
                        author=author,
                        rating=rating,
                        date=review_date,
                        title="",  # OA reviews do not have a separate title
                        review_text=review_text,
                        language="fr",
                    )
                )
        except Exception as exc:
            logger.warning("Opinion Assurances: failed to parse review: %s", exc)

    return reviews


def _build_page_url(page_num: int) -> str:
    """Return the URL for a given page number.

    Page 1 → assureur-alan.html
    Page N → assureur-alan-page{N}.html
    """
    if page_num <= 1:
        return OPINION_ASSURANCES_BASE_URL
    return f"https://www.opinion-assurances.fr/assureur-alan-page{page_num}.html"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_opinion_assurances(
    start_date: date,
    end_date: date,
    max_pages: int = 30,
    progress_callback=None,
) -> List[Review]:
    """Scrape Opinion Assurances reviews for Alan within a date range.

    Handles the site's TLS requirements and pagination scheme
    (``assureur-alan-page{N}.html``).
    """
    all_reviews: List[Review] = []
    session = requests.Session()
    session.mount("https://", _TLSAdapter())
    session.headers.update(COMMON_HEADERS)

    for page_num in range(1, max_pages + 1):
        url = _build_page_url(page_num)
        # The site requires Referer for pages after the first
        if page_num > 1:
            session.headers["Referer"] = _build_page_url(page_num - 1)

        logger.info("Opinion Assurances: fetching page %d – %s", page_num, url)

        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 404:
                logger.info("Opinion Assurances: page %d returned 404, stopping.", page_num)
                break
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Opinion Assurances: request failed on page %d: %s", page_num, exc)
            break

        soup = BeautifulSoup(resp.text, "lxml")

        # Detect bot-block page
        if soup.title and "indisponible" in (soup.title.get_text() or "").lower():
            logger.warning(
                "Opinion Assurances: page %d blocked (anti-bot), "
                "waiting longer and retrying once.",
                page_num,
            )
            polite_sleep(5.0, 8.0)
            try:
                resp = session.get(url, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "lxml")
            except requests.RequestException:
                break

        page_reviews = _extract_reviews_from_page(soup)

        if not page_reviews:
            logger.info("Opinion Assurances: no reviews on page %d, stopping.", page_num)
            break

        all_reviews.extend(page_reviews)

        if progress_callback:
            progress_callback(
                f"Opinion Assurances: page {page_num} → {len(page_reviews)} reviews"
            )

        # Early stop: if oldest review on page is before start_date
        page_dates = []
        for r in page_reviews:
            if r.date:
                try:
                    from datetime import datetime
                    page_dates.append(datetime.strptime(r.date, "%Y-%m-%d").date())
                except ValueError:
                    pass
        if page_dates and min(page_dates) < start_date:
            logger.info(
                "Opinion Assurances: oldest review on page %d (%s) < %s, stopping.",
                page_num, min(page_dates), start_date,
            )
            break

        # Check for next page link in pagination
        pag = soup.select_one(".oa_pagination")
        if pag:
            next_page_url = f"/assureur-alan-page{page_num + 1}.html"
            next_link = pag.select_one(f'a[href="{next_page_url}"]')
            if not next_link:
                logger.info("Opinion Assurances: no page %d link, stopping.", page_num + 1)
                break

        polite_sleep(2.0, 4.0)

    filtered = filter_reviews_by_date(all_reviews, start_date, end_date)
    logger.info("Opinion Assurances: %d reviews after date filtering.", len(filtered))
    return filtered

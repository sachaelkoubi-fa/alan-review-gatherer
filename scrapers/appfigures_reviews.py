"""Client for the Appfigures Reviews API."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import List, Optional, Tuple

import requests

from .base import Review

logger = logging.getLogger(__name__)

APPFIGURES_API_BASE = "https://api.appfigures.com/v2"


def scrape_appfigures(
    start_date: date,
    end_date: date,
    username: str = "",
    password: str = "",
    api_key: str = "",
    products: Optional[str] = None,
    progress_callback=None,
) -> List[Review]:
    """
    Fetch reviews from the Appfigures API within a date range.

    Authentication:
        Appfigures supports HTTP Basic Auth (username + password) with an
        optional ``X-Client-Key`` header for the API client key.

    Parameters
    ----------
    start_date / end_date : date
        Inclusive date range.
    username / password : str
        Appfigures account credentials for Basic Auth.
    api_key : str
        Client API key (sent as ``X-Client-Key``).
    products : str, optional
        Comma-separated product IDs to filter. If *None*, all products are
        returned.
    progress_callback : callable, optional
        Function called with status messages.
    """
    if not username or not password:
        msg = (
            "Appfigures: username and password are required. "
            "Please provide your Appfigures credentials."
        )
        logger.error(msg)
        if progress_callback:
            progress_callback(msg)
        return []

    session = requests.Session()
    session.auth = (username, password)
    if api_key:
        session.headers["X-Client-Key"] = api_key

    all_reviews: List[Review] = []
    page = 1
    page_size = 100  # Appfigures allows up to 500, but 100 is safe

    while True:
        params = {
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d"),
            "count": page_size,
            "page": page,
        }
        if products:
            params["products"] = products

        url = f"{APPFIGURES_API_BASE}/reviews"
        logger.info("Appfigures: fetching page %d – %s", page, url)

        try:
            resp = session.get(url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Appfigures: request failed on page %d: %s", page, exc)
            if progress_callback:
                progress_callback(f"Appfigures: API error – {exc}")
            break

        data = resp.json()

        # The API returns either a dict with "reviews" key or a list directly
        if isinstance(data, dict):
            reviews_list = data.get("reviews", [])
            total = data.get("total", 0)
        elif isinstance(data, list):
            reviews_list = data
            total = None
        else:
            break

        if not reviews_list:
            break

        for item in reviews_list:
            try:
                # Parse the review date
                raw_date = item.get("date", "")
                review_date = ""
                if raw_date:
                    # Appfigures dates can be ISO 8601 or YYYY-MM-DD
                    try:
                        dt = datetime.fromisoformat(
                            raw_date.replace("Z", "+00:00")
                        )
                        review_date = dt.strftime("%Y-%m-%d")
                    except ValueError:
                        review_date = raw_date[:10]  # Fallback: first 10 chars

                # Determine language
                lang = item.get("language", "")
                if not lang:
                    lang = item.get("iso", "unknown")

                # Determine store
                store = item.get("store", "")
                title = item.get("title", "")
                if store:
                    title = f"[{store}] {title}" if title else f"[{store}]"

                all_reviews.append(
                    Review(
                        source="appfigures",
                        author=item.get("author", "Unknown"),
                        rating=float(item.get("stars", 0)),
                        date=review_date,
                        title=title,
                        review_text=item.get("review", ""),
                        language=lang[:2].lower() if lang else "unknown",
                    )
                )
            except Exception as exc:
                logger.warning("Appfigures: failed to parse review: %s", exc)

        if progress_callback:
            progress_callback(
                f"Appfigures: fetched page {page} ({len(reviews_list)} reviews)"
            )

        # Check pagination
        if len(reviews_list) < page_size:
            break
        page += 1

    logger.info("Appfigures: %d total reviews fetched.", len(all_reviews))
    return all_reviews

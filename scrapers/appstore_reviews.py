"""Scraper for Apple App Store and Google Play Store reviews of Alan.

- **Apple App Store**: Uses the public iTunes RSS feed (no auth needed).
  Up to 10 pages × 50 reviews = 500 most-recent reviews.
- **Google Play Store**: Uses the ``google-play-scraper`` library.

No API keys or credentials required.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import date, datetime
from typing import List

from .base import Review, filter_reviews_by_date, polite_sleep

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alan app identifiers
# ---------------------------------------------------------------------------
APPLE_APP_ID = "1277025964"  # Alan France Assurance Santé
GOOGLE_PLAY_PACKAGE = "com.alanmobile"  # Alan France : assurance santé


# ---------------------------------------------------------------------------
# Apple App Store (iTunes RSS feed)
# ---------------------------------------------------------------------------

def _fetch_apple_reviews(
    max_pages: int = 10,
    progress_callback=None,
) -> List[Review]:
    """Fetch reviews from the Apple App Store RSS feed.

    The RSS feed returns up to 50 reviews per page, sorted by most recent.
    Apple provides up to 10 pages (500 reviews max).
    """
    reviews: List[Review] = []
    seen_ids: set[str] = set()

    for page in range(1, max_pages + 1):
        url = (
            f"https://itunes.apple.com/fr/rss/customerreviews/"
            f"page={page}/id={APPLE_APP_ID}/sortby=mostrecent/json"
        )
        logger.info("App Store (Apple): fetching page %d", page)

        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            logger.error("App Store (Apple): failed on page %d: %s", page, exc)
            break

        entries = data.get("feed", {}).get("entry", [])
        if not entries:
            logger.info("App Store (Apple): no entries on page %d, stopping.", page)
            break

        page_count = 0
        for entry in entries:
            # The first entry on page 1 is the app metadata, skip it
            if entry.get("im:name"):
                continue

            entry_id = entry.get("id", {}).get("label", "")
            if entry_id in seen_ids:
                continue
            seen_ids.add(entry_id)

            try:
                author = entry.get("author", {}).get("name", {}).get("label", "Unknown")
                title = entry.get("title", {}).get("label", "")
                rating = float(entry.get("im:rating", {}).get("label", 0))
                content = entry.get("content", {}).get("label", "")
                updated = entry.get("updated", {}).get("label", "")

                # Parse ISO date
                review_date = ""
                if updated:
                    try:
                        dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                        review_date = dt.strftime("%Y-%m-%d")
                    except ValueError:
                        review_date = updated[:10]

                reviews.append(Review(
                    source="app_store_apple",
                    author=author,
                    rating=rating,
                    date=review_date,
                    title=title,
                    review_text=content,
                    language="fr",
                ))
                page_count += 1
            except Exception as exc:
                logger.debug("App Store (Apple): skipped entry: %s", exc)

        logger.info("App Store (Apple): page %d → %d reviews", page, page_count)
        if progress_callback:
            progress_callback(f"App Store (Apple): page {page} → {page_count} reviews")

        if page_count == 0:
            break

        polite_sleep(0.5, 1.5)

    logger.info("App Store (Apple): %d total reviews fetched.", len(reviews))
    return reviews


# ---------------------------------------------------------------------------
# Google Play Store
# ---------------------------------------------------------------------------

def _fetch_play_store_reviews(
    max_reviews: int = 500,
    progress_callback=None,
) -> List[Review]:
    """Fetch reviews from the Google Play Store using google-play-scraper."""
    try:
        from google_play_scraper import reviews as gp_reviews, Sort
    except ImportError:
        logger.error("Google Play: google-play-scraper not installed.")
        return []

    reviews: List[Review] = []
    seen_ids: set[str] = set()

    logger.info("Play Store: fetching reviews for %s", GOOGLE_PLAY_PACKAGE)
    if progress_callback:
        progress_callback("Play Store: fetching reviews…")

    try:
        # google-play-scraper returns up to `count` reviews per call
        # and a continuation token for pagination.
        continuation_token = None
        fetched_total = 0

        while fetched_total < max_reviews:
            batch_size = min(200, max_reviews - fetched_total)
            result, continuation_token = gp_reviews(
                GOOGLE_PLAY_PACKAGE,
                lang="fr",
                country="fr",
                sort=Sort.NEWEST,
                count=batch_size,
                continuation_token=continuation_token,
            )

            if not result:
                break

            for r in result:
                review_id = r.get("reviewId", "")
                if review_id in seen_ids:
                    continue
                seen_ids.add(review_id)

                review_date = ""
                at = r.get("at")
                if at and isinstance(at, datetime):
                    review_date = at.strftime("%Y-%m-%d")

                content = r.get("content", "") or ""
                reviews.append(Review(
                    source="app_store_google_play",
                    author=r.get("userName", "Unknown"),
                    rating=float(r.get("score", 0)),
                    date=review_date,
                    title="",  # Play Store reviews have no title
                    review_text=content,
                    language="fr",
                ))

            fetched_total += len(result)
            logger.info("Play Store: fetched %d reviews so far", fetched_total)
            if progress_callback:
                progress_callback(f"Play Store: {fetched_total} reviews fetched")

            if not continuation_token:
                break

            polite_sleep(0.5, 1.5)

    except Exception as exc:
        logger.error("Play Store: error fetching reviews: %s", exc)

    logger.info("Play Store: %d total reviews fetched.", len(reviews))
    return reviews


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_app_store_reviews(
    start_date: date,
    end_date: date,
    include_apple: bool = True,
    include_play_store: bool = True,
    max_reviews_per_store: int = 500,
    progress_callback=None,
) -> List[Review]:
    """Scrape app store reviews for Alan within a date range.

    Combines Apple App Store (iTunes RSS) and Google Play Store reviews
    into a single list, filtered by date.
    """
    all_reviews: List[Review] = []

    if include_apple:
        apple_reviews = _fetch_apple_reviews(
            max_pages=10,
            progress_callback=progress_callback,
        )
        all_reviews.extend(apple_reviews)

    if include_play_store:
        play_reviews = _fetch_play_store_reviews(
            max_reviews=max_reviews_per_store,
            progress_callback=progress_callback,
        )
        all_reviews.extend(play_reviews)

    filtered = filter_reviews_by_date(all_reviews, start_date, end_date)
    logger.info(
        "App Stores: %d reviews after date filtering (Apple: %d, Play: %d → filtered: %d).",
        len(all_reviews),
        sum(1 for r in all_reviews if r.source == "app_store_apple"),
        sum(1 for r in all_reviews if r.source == "app_store_google_play"),
        len(filtered),
    )
    return filtered

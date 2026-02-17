"""Base review dataclass and shared utilities for all scrapers."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, asdict, fields
from datetime import date, datetime
from typing import List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Unified Review data model
# ---------------------------------------------------------------------------

@dataclass
class Review:
    """Normalized review from any source."""

    source: str  # google | appfigures | trustpilot | opinion_assurances
    author: str
    rating: float  # 1-5 scale
    date: str  # YYYY-MM-DD
    title: str
    review_text: str
    language: str  # fr / en / unknown

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def column_order() -> List[str]:
        """Return the canonical column order for CSV export."""
        return [f.name for f in fields(Review)]


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def reviews_to_dataframe(reviews: List[Review]) -> pd.DataFrame:
    """Convert a list of Review objects to a pandas DataFrame."""
    if not reviews:
        return pd.DataFrame(columns=Review.column_order())
    df = pd.DataFrame([asdict(r) for r in reviews])
    return df[Review.column_order()]


def filter_reviews_by_date(
    reviews: List[Review],
    start_date: date,
    end_date: date,
) -> List[Review]:
    """Keep only reviews whose date falls within [start_date, end_date]."""
    filtered: List[Review] = []
    for r in reviews:
        try:
            review_date = datetime.strptime(r.date, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        if start_date <= review_date <= end_date:
            filtered.append(r)
    return filtered


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

COMMON_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def polite_sleep(min_sec: float = 1.0, max_sec: float = 3.0) -> None:
    """Sleep a random duration to avoid rate-limiting."""
    time.sleep(random.uniform(min_sec, max_sec))

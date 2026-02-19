"""Scraper for Google Reviews of Alan.

These are the same reviews visible on Google Maps at:
  https://www.google.com/maps/place/Alan/@48.8754333,2.3632119

Google Maps blocks review content for non-authenticated headless browsers
("affichage limité"), so we scrape them via the Google Search reviews panel
which shows the identical Google Business Profile reviews.

Requires Playwright (with system Chrome when available).
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import List, Optional

from dateutil.relativedelta import relativedelta

from .base import Review, filter_reviews_by_date

logger = logging.getLogger(__name__)

# Google Maps page for Alan (for reference — same reviews)
GOOGLE_MAPS_URL = (
    "https://www.google.com/maps/place/Alan/"
    "@48.8754333,2.360637,2068m/"
    "data=!3m1!1e3!4m8!3m7!1s0x47e66e14e850c89f:0x851109550d1a4c7"
    "!8m2!3d48.8754333!4d2.3632119!9m1!1b1"
    "!16s%2Fg%2F11csptw_ww?entry=ttu"
)

# Google Search reviews URL for Alan — this displays the Google Business
# Profile reviews in a scrollable panel (same reviews as on Google Maps).
# The `si` parameter is the stable business identifier.
GOOGLE_REVIEWS_URL = (
    "https://www.google.com/search"
    "?q=Alan+Reviews"
    "&si=AL3DRZEsmMGCryMMFSHJ3StBhOdZ2-6yYkXd_doETEE1OR-qOZvqdoj8ZRNdB3TtWrGcbzTg7lbKcf5gmiSM2SrfLW4hcufvAETXJQByNOEI-4VZTDx1B2XVeJJR7jFnzVEQRXMijKcd"
    "&hl=fr"
)


# ---------------------------------------------------------------------------
# Relative-date parsing  ("il y a 3 semaines" → date)
# ---------------------------------------------------------------------------

_FR_UNITS = {
    "seconde": "seconds",
    "secondes": "seconds",
    "minute": "minutes",
    "minutes": "minutes",
    "heure": "hours",
    "heures": "hours",
    "jour": "days",
    "jours": "days",
    "semaine": "weeks",
    "semaines": "weeks",
    "mois": "months",
    "an": "years",
    "ans": "years",
}

_EN_UNITS = {
    "second": "seconds",
    "seconds": "seconds",
    "minute": "minutes",
    "minutes": "minutes",
    "hour": "hours",
    "hours": "hours",
    "day": "days",
    "days": "days",
    "week": "weeks",
    "weeks": "weeks",
    "month": "months",
    "months": "months",
    "year": "years",
    "years": "years",
}


def _parse_relative_date(text: str) -> Optional[str]:
    """Convert a relative date string to YYYY-MM-DD.

    Handles French ("il y a 3 semaines") and English ("3 weeks ago") formats,
    as well as "un/une" (= 1).
    """
    text = text.strip().lower()
    text = text.replace("\xa0", " ")  # non-breaking spaces

    # "il y a un mois", "il y a 3 semaines", "il y a une heure"
    m = re.search(r"il y a\s+(\d+|un|une)\s+(\w+)", text)
    if not m:
        # English fallback
        m = re.search(r"(\d+|a|an)\s+(\w+)\s+ago", text)
        if m:
            units_map = _EN_UNITS
        else:
            return None
    else:
        units_map = _FR_UNITS

    raw_num, raw_unit = m.group(1), m.group(2)
    num = 1 if raw_num in ("un", "une", "a", "an") else int(raw_num)
    unit = units_map.get(raw_unit)
    if not unit:
        return None

    now = datetime.now()
    if unit == "weeks":
        delta = relativedelta(weeks=num)
    elif unit == "months":
        delta = relativedelta(months=num)
    elif unit == "years":
        delta = relativedelta(years=num)
    elif unit == "days":
        delta = relativedelta(days=num)
    elif unit == "hours":
        delta = relativedelta(hours=num)
    elif unit == "minutes":
        delta = relativedelta(minutes=num)
    elif unit == "seconds":
        delta = relativedelta(seconds=num)
    else:
        return None

    return (now - delta).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# JavaScript to extract reviews from the DOM
# ---------------------------------------------------------------------------

_JS_EXTRACT_REVIEWS = r"""() => {
    const cards = document.querySelectorAll('div.SEzcUb');
    const reviews = [];

    for (const card of cards) {
        const text = card.innerText || '';
        if (!text.includes('Avis de')) continue;

        const lines = text.split('\n').map(l => l.trim()).filter(l => l);
        if (lines.length < 3) continue;

        const author = lines[0] || '';
        let rating = 0;
        let dateLine = '';
        let bodyStart = 3;

        for (let i = 1; i < Math.min(8, lines.length); i++) {
            // Match "X/5" rating (standalone or with date)
            const standalone = lines[i].match(/^([1-5])\/5$/);
            if (standalone) {
                rating = parseInt(standalone[1]);
                // Date is usually on same or next line after " · "
                const nextLine = (lines[i + 1] || '').replace(/^\s*·?\s*/, '');
                if (nextLine.includes('il y a') || nextLine.includes('ago')) {
                    dateLine = nextLine;
                    bodyStart = i + 2;
                } else {
                    bodyStart = i + 1;
                }
                break;
            }
            // "X/5 · il y a 3 semaines" combined
            const combined = lines[i].match(/^([1-5])\/5\s*·\s*(.+)/);
            if (combined) {
                rating = parseInt(combined[1]);
                dateLine = combined[2];
                bodyStart = i + 1;
                break;
            }
        }

        // Extract body text, skipping "NOUVEAU", "Plus", "Visité en ..."
        const bodyLines = lines.slice(bodyStart).filter(l =>
            l !== 'NOUVEAU' && l !== 'Plus' && l !== '…Plus' && !l.startsWith('Visité en')
        );
        const body = bodyLines.join(' ').replace(/…Plus$/, '').trim();

        if (author && rating > 0) {
            reviews.push({ author, rating, dateLine, body });
        }
    }
    return reviews;
}"""


# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------

async def _scrape_google_reviews(
    start_date: date,
    end_date: date,
    max_reviews: int = 500,
    progress_callback=None,
) -> List[Review]:
    """Scrape Google reviews from the Google Search reviews panel.

    These are the same reviews displayed on Google Maps for Alan.
    """
    from playwright.async_api import async_playwright

    reviews: List[Review] = []
    raw_count = 0

    async with async_playwright() as pw:
        # Try system Chrome first, fall back to bundled Chromium
        launch_kwargs: dict = {
            "headless": True,
            "args": [
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        }
        try:
            browser = await pw.chromium.launch(channel="chrome", **launch_kwargs)
            logger.info("Google Reviews: using system Chrome.")
        except Exception:
            logger.info("Google Reviews: system Chrome not found, using bundled Chromium.")
            browser = await pw.chromium.launch(**launch_kwargs)

        context = await browser.new_context(
            locale="fr-FR",
            viewport={"width": 1470, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/133.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()
        await page.add_init_script(
            'Object.defineProperty(navigator, "webdriver", { get: () => undefined });'
        )

        try:
            if progress_callback:
                progress_callback("Google Reviews: loading page…")

            logger.info("Google Reviews: navigating to Google Search reviews panel")
            await page.goto(GOOGLE_REVIEWS_URL, timeout=30000)
            await page.wait_for_timeout(4000)

            # Check for CAPTCHA / sorry page
            if "sorry" in page.url:
                logger.error("Google Reviews: blocked by CAPTCHA. Try again later.")
                if progress_callback:
                    progress_callback("Google Reviews: blocked by Google, retrying…")
                await browser.close()
                return []

            # Accept cookies
            try:
                btn = page.locator("button:has-text('Tout accepter')").first
                if await btn.is_visible(timeout=4000):
                    await btn.click()
                    logger.info("Google Reviews: cookie consent accepted.")
                    await page.wait_for_timeout(2000)
            except Exception:
                pass

            # Click "Autres avis d'utilisateurs" to expand the full review list
            try:
                autres = page.locator("text=Autres avis").first
                if await autres.is_visible(timeout=3000):
                    await autres.click()
                    logger.info("Google Reviews: clicked 'Autres avis' to expand.")
                    await page.wait_for_timeout(3000)
                else:
                    logger.info("Google Reviews: 'Autres avis' not found, reviews may already be expanded.")
            except Exception:
                logger.info("Google Reviews: could not click 'Autres avis'.")

            if progress_callback:
                progress_callback("Google Reviews: scrolling to load all reviews…")

            # Scroll down to load more reviews (Google lazy-loads them)
            prev_count = 0
            no_change_rounds = 0
            for scroll_i in range(30):
                await page.evaluate("window.scrollBy(0, 2000)")
                await page.wait_for_timeout(1500)

                count = await page.locator("div.SEzcUb").count()
                logger.info("Google Reviews: scroll %d → %d review cards visible", scroll_i + 1, count)

                if count == prev_count:
                    no_change_rounds += 1
                    if no_change_rounds >= 3:
                        logger.info("Google Reviews: no new reviews after 3 scrolls, stopping.")
                        break
                else:
                    no_change_rounds = 0
                prev_count = count

                if count >= max_reviews:
                    break

            # Extract all reviews from the DOM
            raw_reviews = await page.evaluate(_JS_EXTRACT_REVIEWS)
            raw_count = len(raw_reviews)
            logger.info("Google Reviews: extracted %d raw reviews from DOM.", raw_count)

            if progress_callback:
                progress_callback(f"Google Reviews: {raw_count} reviews extracted, filtering by date…")

            # Convert to Review objects with date parsing
            seen: set = set()
            for r in raw_reviews:
                author = r.get("author", "")
                rating = float(r.get("rating", 0))
                date_str = _parse_relative_date(r.get("dateLine", "")) or ""
                body = r.get("body", "")

                # Deduplicate by author + body prefix
                dedup_key = f"{author}|{body[:80]}"
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                reviews.append(Review(
                    source="google",
                    author=author,
                    rating=rating,
                    date=date_str,
                    title="",
                    review_text=body,
                    language="fr",
                ))

        except Exception as exc:
            logger.error("Google Reviews: error during scraping: %s", exc)
        finally:
            await browser.close()

    filtered = filter_reviews_by_date(reviews, start_date, end_date)
    logger.info(
        "Google Reviews: %d raw → %d after dedup → %d after date filter.",
        raw_count,
        len(reviews),
        len(filtered),
    )
    return filtered


# ---------------------------------------------------------------------------
# Public synchronous wrapper
# ---------------------------------------------------------------------------

def scrape_google_reviews(
    start_date: date,
    end_date: date,
    max_reviews: int = 500,
    progress_callback=None,
) -> List[Review]:
    """Synchronous entry point for the Google Reviews scraper.

    Scrapes Google Business Profile reviews for Alan — the same reviews
    visible at: https://www.google.com/maps/place/Alan/
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(
                    asyncio.run,
                    _scrape_google_reviews(
                        start_date, end_date, max_reviews, progress_callback
                    ),
                ).result()
    except RuntimeError:
        pass

    return asyncio.run(
        _scrape_google_reviews(
            start_date, end_date, max_reviews, progress_callback
        )
    )

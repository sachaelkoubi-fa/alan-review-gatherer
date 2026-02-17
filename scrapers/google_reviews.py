"""Scraper for Google Maps / Business reviews of Alan using Playwright.

Uses ``dateutil.relativedelta`` for accurate relative-date parsing
(e.g. "il y a 2 mois" → exactly 2 calendar months back, not 60 days).
"""

from __future__ import annotations

import logging
import os
import re
from datetime import date, datetime, timedelta
from typing import List, Optional

from dateutil.relativedelta import relativedelta

from .base import Review, filter_reviews_by_date

logger = logging.getLogger(__name__)

# Google Maps Place ID URL for Alan (more reliable than search).
# Falls back to search-based URL if the place ID link fails.
GOOGLE_MAPS_PLACE_URL = (
    "https://www.google.com/maps/place/Alan/@48.8566,2.3522,12z/"
)
GOOGLE_MAPS_SEARCH_URL = (
    "https://www.google.com/maps/search/Alan+assurance+santé+france"
)

# ---------------------------------------------------------------------------
# Relative-date parsing ("il y a 2 mois", "3 months ago", etc.)
# ---------------------------------------------------------------------------

# Maps a French/English time-unit word to a relativedelta keyword.
_UNIT_TO_KWARG: dict[str, str] = {
    # French
    "jour": "days",
    "jours": "days",
    "semaine": "weeks",
    "semaines": "weeks",
    "mois": "months",
    "an": "years",
    "ans": "years",
    # English
    "day": "days",
    "days": "days",
    "week": "weeks",
    "weeks": "weeks",
    "month": "months",
    "months": "months",
    "year": "years",
    "years": "years",
}


def _parse_relative_date(text: str, reference: date | None = None) -> Optional[str]:
    """Convert a relative date string to YYYY-MM-DD using calendar-accurate arithmetic.

    Examples:
        "il y a 2 mois"  → exactly 2 calendar months before *reference*
        "3 months ago"    → exactly 3 calendar months before *reference*
        "il y a un an"    → exactly 1 year before *reference*
    """
    if reference is None:
        reference = date.today()
    text = text.strip().lower()

    # French: "il y a X <unit>" or "il y a un/une <unit>"
    fr_match = re.search(r"il y a\s+(\d+|un|une)\s+(\w+)", text)
    if fr_match:
        count_str, unit = fr_match.groups()
        count = 1 if count_str in ("un", "une") else int(count_str)
        kwarg = _UNIT_TO_KWARG.get(unit)
        if kwarg:
            delta = relativedelta(**{kwarg: count})
            return (reference - delta).strftime("%Y-%m-%d")

    # English: "X <unit> ago" or "a/an <unit> ago"
    en_match = re.search(r"(\d+|a|an)\s+(\w+)\s+ago", text)
    if en_match:
        count_str, unit = en_match.groups()
        count = 1 if count_str in ("a", "an") else int(count_str)
        kwarg = _UNIT_TO_KWARG.get(unit)
        if kwarg:
            delta = relativedelta(**{kwarg: count})
            return (reference - delta).strftime("%Y-%m-%d")

    return None


def _parse_star_rating(text: str) -> float:
    """Extract rating from text like '4 étoiles sur 5' or '4/5' or aria-label."""
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:étoile|star|/\s*5|sur\s*5)", text.lower())
    if m:
        return float(m.group(1).replace(",", "."))
    # Try just a bare number 1-5
    m2 = re.search(r"\b([1-5])\b", text)
    if m2:
        return float(m2.group(1))
    return 0.0


async def _accept_cookies(page) -> None:
    """Dismiss the Google cookie-consent dialog if visible."""
    try:
        consent_btn = page.locator(
            "button:has-text('Tout accepter'), "
            "button:has-text('Accept all'), "
            "form[action*='consent'] button"
        ).first
        if await consent_btn.is_visible(timeout=4000):
            await consent_btn.click()
            logger.info("Google Reviews: cookie consent accepted.")
            await page.wait_for_timeout(2000)
        else:
            logger.info("Google Reviews: no cookie consent dialog detected.")
    except Exception:
        logger.info("Google Reviews: cookie consent check skipped (not found).")


async def _dismiss_popups(page) -> None:
    """Close any informational popups/tooltips that may overlay the page.

    IMPORTANT: We must NOT click the panel's own 'Fermer'/'Close' button
    because that closes the entire place panel, losing our context.
    We only press Escape, which dismisses overlays/tooltips without
    closing the place panel.
    """
    try:
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(500)
        logger.info("Google Reviews: pressed Escape to dismiss any overlay.")
    except Exception:
        pass


async def _navigate_to_reviews(page) -> bool:
    """Navigate from search results → place → reviews panel.

    Returns True if we likely reached the reviews panel.
    """
    current_url = page.url
    logger.info("Google Reviews: current URL after load: %s", current_url)

    # If we're on a search results page, click the first result
    if "/maps/search/" in current_url or "/maps/place/" not in current_url:
        try:
            first_result = page.locator("a[href*='/maps/place/']").first
            if await first_result.is_visible(timeout=5000):
                await first_result.click()
                await page.wait_for_timeout(3000)
                logger.info("Google Reviews: clicked first search result → %s", page.url)
            else:
                logger.info("Google Reviews: no search result link — place may have auto-opened.")
        except Exception as exc:
            logger.warning("Google Reviews: could not click first search result: %s", exc)

    # Dismiss any popups (e.g. "Google ne vérifie pas les avis")
    await _dismiss_popups(page)

    # Strategy: open the reviews panel.
    # Google Maps layouts vary – some places have an "Avis" tab, others
    # only expose reviews through the "Présentation" tab. We try several
    # approaches in order of reliability.
    reviews_opened = False

    # Try 1: Wait for the "Avis" / "Reviews" tab to appear.
    # Google Maps renders tabs asynchronously – it can take 5-15s depending
    # on the page load and JavaScript execution speed.  We poll repeatedly
    # instead of using a single timeout.
    for wait_round in range(6):  # up to 6 × 2s = 12s total
        for tab_sel in [
            "button[role='tab']:has-text('Avis')",
            "button[role='tab']:has-text('Reviews')",
        ]:
            try:
                tab = page.locator(tab_sel).first
                if await tab.is_visible(timeout=2000):
                    tab_text = (await tab.inner_text(timeout=1000)).strip()
                    if "avis" in tab_text.lower() or "review" in tab_text.lower():
                        await tab.click()
                        await page.wait_for_timeout(3000)
                        logger.info("Google Reviews: clicked tab '%s' (wait round %d).", tab_text, wait_round)
                        reviews_opened = True
                        break
            except Exception:
                continue
        if reviews_opened:
            break
        logger.info("Google Reviews: Avis tab not found (round %d/6), waiting...", wait_round + 1)
        await page.wait_for_timeout(1000)

    # Try 2: Scroll the side panel to find the reviews section, then click
    # a review-count link ("Tous les avis") to open the full reviews list.
    # The Alan place page has the reviews BELOW the fold in the Présentation tab.
    # NOTE: we avoid `button[aria-label*='avis']` because it matches the info
    # icon (ⓘ) next to the rating, which opens a tooltip instead.
    if not reviews_opened:
        logger.info("Google Reviews: no Avis tab found – scrolling panel to find review section...")
        scroll_container = page.locator(
            "div[role='main'] div.m6QErb"
        ).first
        try:
            for scroll_step in range(10):
                await scroll_container.evaluate("el => el.scrollTop += 400")
                await page.wait_for_timeout(800)

                # Check if review cards appeared inline
                card_count = await page.locator("div.jftiEf, div[data-review-id]").count()
                if card_count > 0:
                    logger.info(
                        "Google Reviews: found %d inline review cards after scrolling (step %d).",
                        card_count, scroll_step,
                    )
                    reviews_opened = True
                    break

                # Look for specific review-section links
                for link_sel in [
                    "button[jsaction*='review']",
                    "button.HHrUdb",
                    "a[href*='lrd']",
                    "text=/Tous les avis/i",
                    "text=/Voir plus d.avis/i",
                    "text=/See all.*review/i",
                ]:
                    try:
                        el = page.locator(link_sel).first
                        if await el.is_visible(timeout=400):
                            el_text = ""
                            try:
                                el_text = (await el.inner_text(timeout=400)).strip()[:60]
                            except Exception:
                                pass
                            await el.click()
                            await page.wait_for_timeout(3000)
                            logger.info(
                                "Google Reviews: scrolled & clicked '%s' (text: '%s') at step %d.",
                                link_sel, el_text, scroll_step,
                            )
                            reviews_opened = True
                            break
                    except Exception:
                        continue
                if reviews_opened:
                    break
        except Exception as exc:
            logger.debug("Google Reviews: scroll error: %s", exc)

    # Try 3: Click the "N avis Google" text in the header area
    if not reviews_opened:
        try:
            avis_text = page.locator("text=/\\d+\\s*avis\\s*(Google)?/i").first
            if await avis_text.is_visible(timeout=2000):
                await avis_text.click()
                await page.wait_for_timeout(3000)
                logger.info("Google Reviews: clicked 'N avis' text.")
                reviews_opened = True
        except Exception:
            pass

    # Try 4: Navigate directly to the place reviews URL
    if not reviews_opened:
        logger.info("Google Reviews: fallback – reloading with direct place search + /reviews")
        try:
            # Try appending review tab to the current place URL
            current = page.url
            if "/maps/place/" in current:
                # Strip existing query parameters and add /reviews
                base = current.split("?")[0].split("/data=")[0]
                if not base.endswith("/"):
                    base += "/"
                reviews_url = base + "reviews"
                await page.goto(reviews_url, timeout=15000)
                await page.wait_for_timeout(4000)
                await _dismiss_popups(page)
                card_count = await page.locator("div.jftiEf, div[data-review-id]").count()
                if card_count > 0:
                    logger.info("Google Reviews: found %d reviews via direct URL.", card_count)
                    reviews_opened = True
        except Exception as exc:
            logger.debug("Google Reviews: direct URL fallback failed: %s", exc)

    if not reviews_opened:
        logger.warning("Google Reviews: could not find reviews tab/link to click!")

    # Wait for review cards to appear (they load dynamically)
    logger.info("Google Reviews: waiting for review cards to render...")
    for wait_iter in range(5):
        for sel in ["div.jftiEf", "div[data-review-id]"]:
            count = await page.locator(sel).count()
            if count > 0:
                logger.info("Google Reviews: found %d review cards (%s) after %d wait(s).", count, sel, wait_iter)
                break
        else:
            await page.wait_for_timeout(2000)
            continue
        break
    else:
        logger.warning("Google Reviews: review cards did not appear after waiting.")

    # Sort by newest first
    try:
        sort_btn = page.locator(
            "button[aria-label*='Trier'], "
            "button[aria-label*='Sort'], "
            "button[data-value='Trier']"
        ).first
        if await sort_btn.is_visible(timeout=3000):
            await sort_btn.click()
            await page.wait_for_timeout(1000)
            newest_option = page.locator(
                "[data-index='1'], "
                "li:has-text('Plus récents'), "
                "li:has-text('Newest')"
            ).first
            if await newest_option.is_visible(timeout=2000):
                await newest_option.click()
                await page.wait_for_timeout(2000)
                logger.info("Google Reviews: sorted by newest.")
    except Exception:
        logger.info("Google Reviews: could not sort by newest.")

    return True


async def _scrape_with_playwright(
    start_date: date,
    end_date: date,
    max_reviews: int = 500,
    progress_callback=None,
) -> List[Review]:
    """Internal async implementation using Playwright."""
    from playwright.async_api import async_playwright

    reviews: List[Review] = []

    async with async_playwright() as pw:
        # Try system Chrome first (more stealth), fall back to bundled Chromium.
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
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/133.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        # Anti-detection: remove the "webdriver" flag
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.chrome = { runtime: {} };
        """)

        try:
            if progress_callback:
                progress_callback("Google Reviews: loading Google Maps...")

            # Navigate to the search results
            logger.info("Google Reviews: navigating to %s", GOOGLE_MAPS_SEARCH_URL)
            await page.goto(GOOGLE_MAPS_SEARCH_URL, timeout=30000)
            await page.wait_for_timeout(3000)

            # Accept cookies if prompted
            await _accept_cookies(page)

            # Navigate to the reviews
            await _navigate_to_reviews(page)

            # --- Detect the review cards container ---
            # Try multiple selectors for the scrollable reviews panel
            scrollable = None
            for sel in [
                "div.m6QErb.DxyBCb.kA9KIf.dS8AEf",
                "div[role='main'] div.m6QErb",
                "div.m6QErb",
            ]:
                loc = page.locator(sel).first
                try:
                    if await loc.is_visible(timeout=2000):
                        scrollable = loc
                        logger.info("Google Reviews: found scrollable container with selector: %s", sel)
                        break
                except Exception:
                    continue

            if not scrollable:
                logger.warning("Google Reviews: could not find scrollable reviews container!")
                # Save debug screenshot
                try:
                    screenshot_path = os.path.join(os.path.dirname(__file__), "..", "debug_google.png")
                    await page.screenshot(path=screenshot_path)
                    logger.warning("Google Reviews: saved debug screenshot to %s", screenshot_path)
                except Exception:
                    pass
                # Log page title and URL for debugging
                title = await page.title()
                logger.warning("Google Reviews: page title = '%s', URL = '%s'", title, page.url)
                return reviews

            # --- Detect review card selectors ---
            # Google Maps uses different class names; try several
            review_card_selector = None
            for sel in [
                "div.jftiEf",
                "div[data-review-id]",
                "div.GHT2ce",
                "div[jscontroller] div.jftiEf",
            ]:
                count = await page.locator(sel).count()
                logger.info("Google Reviews: selector '%s' matched %d elements", sel, count)
                if count > 0 and review_card_selector is None:
                    review_card_selector = sel

            if not review_card_selector:
                logger.warning("Google Reviews: no review card selector matched any elements!")
                # Save debug info
                try:
                    screenshot_path = os.path.join(os.path.dirname(__file__), "..", "debug_google.png")
                    await page.screenshot(path=screenshot_path)
                    logger.warning("Google Reviews: saved debug screenshot to %s", screenshot_path)
                    # Also log a snippet of the page HTML
                    body_html = await page.locator("body").inner_html()
                    logger.info("Google Reviews: page body snippet (first 2000 chars): %s", body_html[:2000])
                except Exception:
                    pass
                return reviews

            logger.info("Google Reviews: using review card selector: '%s'", review_card_selector)

            # Track already-seen reviews to prevent duplicates.
            seen_keys: set[str] = set()
            no_new_count = 0

            for scroll_iter in range(100):  # Safety limit
                review_elements = await page.locator(review_card_selector).all()

                if scroll_iter == 0:
                    logger.info("Google Reviews: initial element count: %d", len(review_elements))

                new_this_round = 0
                for elem in review_elements:
                    try:
                        # --- Author ---
                        author_el = elem.locator(
                            ".d4r55, .WNxzHc [class*='fontTitleSmall']"
                        ).first
                        author = ""
                        if await author_el.count():
                            author = (await author_el.inner_text(timeout=1000)).strip()
                        if not author:
                            author = "Unknown"

                        # --- Rating ---
                        star_el = elem.locator(
                            "span[role='img'][aria-label], .kvMYJc"
                        ).first
                        rating = 0.0
                        if await star_el.count():
                            aria = await star_el.get_attribute("aria-label") or ""
                            rating = _parse_star_rating(aria)

                        # --- Date ---
                        date_el = elem.locator(
                            ".rsqaWe, .DU9Pgb"
                        ).first
                        review_date = ""
                        if await date_el.count():
                            date_text = await date_el.inner_text(timeout=1000)
                            parsed = _parse_relative_date(date_text)
                            if parsed:
                                review_date = parsed

                        # --- Expand "More" text ---
                        try:
                            more_btn = elem.locator(
                                "button.w8nwRe, button:has-text('Plus'), button:has-text('More')"
                            ).first
                            if await more_btn.is_visible(timeout=500):
                                await more_btn.click()
                                await page.wait_for_timeout(300)
                        except Exception:
                            pass

                        # --- Review text ---
                        text_el = elem.locator(
                            ".wiI7pd, .MyEned span"
                        ).first
                        review_text = ""
                        if await text_el.count():
                            review_text = (await text_el.inner_text(timeout=1000)).strip()

                        if review_date or review_text:
                            # Prefer data-review-id if available; otherwise
                            # content-based key (author + date + first 120 chars).
                            review_id = await elem.get_attribute("data-review-id")
                            dedup_key = review_id or f"{author}|{review_date}|{review_text[:120]}"
                            if dedup_key in seen_keys:
                                continue
                            seen_keys.add(dedup_key)

                            reviews.append(
                                Review(
                                    source="google",
                                    author=author,
                                    rating=rating,
                                    date=review_date,
                                    title="",
                                    review_text=review_text,
                                    language="fr",
                                )
                            )
                            new_this_round += 1

                    except Exception as exc:
                        logger.debug("Google Reviews: error parsing element: %s", exc)
                        continue

                logger.info(
                    "Google Reviews: scroll %d — %d elements found, %d new, %d total unique",
                    scroll_iter + 1, len(review_elements), new_this_round, len(reviews),
                )
                if progress_callback:
                    progress_callback(
                        f"Google Reviews: collected {len(reviews)} unique reviews (scroll {scroll_iter + 1})"
                    )

                if new_this_round == 0:
                    no_new_count += 1
                    if no_new_count >= 3:
                        logger.info("Google Reviews: 3 consecutive scrolls with no new reviews, stopping.")
                        break
                else:
                    no_new_count = 0

                if len(reviews) >= max_reviews:
                    break

                # Check if oldest review is already past our date range
                if reviews:
                    dates = [
                        datetime.strptime(r.date, "%Y-%m-%d").date()
                        for r in reviews
                        if r.date
                    ]
                    if dates and min(dates) < start_date:
                        logger.info("Google Reviews: reached reviews older than start_date, stopping scroll.")
                        break

                # Scroll down
                try:
                    await scrollable.evaluate(
                        "el => el.scrollTop = el.scrollHeight"
                    )
                except Exception:
                    # Fallback: scroll the page
                    await page.mouse.wheel(0, 3000)
                await page.wait_for_timeout(2000)

        except Exception as exc:
            logger.error("Google Reviews: unexpected error: %s", exc)
            if progress_callback:
                progress_callback(f"Google Reviews: error – {exc}")
        finally:
            await browser.close()

    return reviews


def _run_async(coro):
    """Run an async coroutine, handling nested event loops (e.g. Streamlit)."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def scrape_google_reviews(
    start_date: date,
    end_date: date,
    max_reviews: int = 500,
    progress_callback=None,
    max_retries: int = 3,
) -> List[Review]:
    """
    Scrape Google Maps reviews for Alan within a date range.

    Uses Playwright (headless Chromium) to handle the dynamically-loaded
    review feed.  Retries up to *max_retries* times if no reviews are found
    (Google Maps rendering is non-deterministic).
    """
    for attempt in range(1, max_retries + 1):
        reviews = _run_async(
            _scrape_with_playwright(start_date, end_date, max_reviews, progress_callback)
        )

        filtered = filter_reviews_by_date(reviews, start_date, end_date)
        logger.info(
            "Google Reviews: attempt %d/%d — %d reviews after date filtering.",
            attempt, max_retries, len(filtered),
        )

        if filtered:
            return filtered

        if attempt < max_retries:
            logger.info("Google Reviews: retrying (attempt %d/%d)...", attempt + 1, max_retries)
            if progress_callback:
                progress_callback(f"Google Reviews: retrying ({attempt + 1}/{max_retries})...")

    logger.warning("Google Reviews: all %d attempts returned 0 reviews.", max_retries)
    return []

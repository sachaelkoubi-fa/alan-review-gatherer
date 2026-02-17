"""
Alan Review Gatherer – Streamlit Web App
=========================================
Collects Alan reviews from Google Reviews, Trustpilot, Opinion Assurances,
App Store & Play Store within a user-specified date range, and exports them
as a unified CSV file.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Callable, Dict, List, Tuple

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Ensure Playwright browsers are installed (needed for cloud deployment)
# ---------------------------------------------------------------------------
@st.cache_resource
def _install_playwright_browsers():
    """Install Playwright Chromium browser if not already present."""
    try:
        subprocess.run(
            ["playwright", "install", "chromium"],
            check=True,
            capture_output=True,
            timeout=120,
        )
    except Exception:
        pass  # Will fail gracefully at scrape time if browsers are missing


_install_playwright_browsers()

from scrapers.base import Review, reviews_to_dataframe
from scrapers.trustpilot_reviews import scrape_trustpilot
from scrapers.opinion_assurances_reviews import scrape_opinion_assurances
from scrapers.google_reviews import scrape_google_reviews
from scrapers.appstore_reviews import scrape_app_store_reviews

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Alan Review Gatherer",
    page_icon="📋",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main container */
    .main .block-container { max-width: 1200px; padding-top: 1.5rem; }

    /* Progress bar color */
    .stProgress > div > div > div { background-color: #4F46E5; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 14px 18px;
        border: 1px solid #e9ecef;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label {
        color: #6b7280;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
    }

    /* Download button */
    div.stDownloadButton > button {
        background-color: #4F46E5;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
    }
    div.stDownloadButton > button:hover {
        background-color: #4338CA;
        color: white;
    }

    /* Smaller caption inside metric delta */
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 0.82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------
SOURCE_LABELS: Dict[str, str] = {
    "trustpilot": "Trustpilot",
    "opinion_assurances": "Opinion Assurances",
    "google": "Google Reviews",
    "app_store_apple": "App Store (iOS)",
    "app_store_google_play": "Play Store (Android)",
}

SOURCE_ICONS: Dict[str, str] = {
    "trustpilot": "⭐",
    "opinion_assurances": "🛡️",
    "google": "🔍",
    "app_store_apple": "🍎",
    "app_store_google_play": "🤖",
}

SOURCE_COLORS: Dict[str, str] = {
    "trustpilot": "#00b67a",
    "opinion_assurances": "#f59e0b",
    "google": "#4285f4",
    "app_store_apple": "#007AFF",
    "app_store_google_play": "#34A853",
}


def _friendly_source(key: str) -> str:
    """Return a human-readable source label."""
    return SOURCE_LABELS.get(key, key)


# ---------------------------------------------------------------------------
# Sidebar – configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📋 Alan Review Gatherer")
    st.caption("Collect & export reviews from multiple platforms")

    st.divider()

    st.subheader("📅 Date Range")
    col_s, col_e = st.columns(2)
    with col_s:
        default_start = date.today() - timedelta(days=90)
        start_date = st.date_input("From", value=default_start)
    with col_e:
        end_date = st.date_input("To", value=date.today())

    if start_date > end_date:
        st.error("⚠️ Start date must be before end date.")

    st.divider()

    st.subheader("📡 Sources")
    use_trustpilot = st.checkbox("⭐ Trustpilot", value=True)
    use_opinion_assurances = st.checkbox("🛡️ Opinion Assurances", value=True)
    use_google = st.checkbox("🔍 Google Reviews", value=True)
    use_app_stores = st.checkbox("📱 App Stores (iOS + Android)", value=True)

    st.divider()

    if st.button("🔄 Clear Cache", help="Clear cached scraping results"):
        st.cache_data.clear()
        st.toast("Cache cleared!", icon="✅")


# ---------------------------------------------------------------------------
# Cached scrape functions (TTL = 15 min)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def _cached_trustpilot(start_str: str, end_str: str) -> Tuple[List[dict], str]:
    try:
        sd = datetime.strptime(start_str, "%Y-%m-%d").date()
        ed = datetime.strptime(end_str, "%Y-%m-%d").date()
        reviews = scrape_trustpilot(sd, ed)
        return [r.__dict__ for r in reviews], ""
    except Exception as exc:
        return [], str(exc)


@st.cache_data(ttl=900, show_spinner=False)
def _cached_opinion_assurances(start_str: str, end_str: str) -> Tuple[List[dict], str]:
    try:
        sd = datetime.strptime(start_str, "%Y-%m-%d").date()
        ed = datetime.strptime(end_str, "%Y-%m-%d").date()
        reviews = scrape_opinion_assurances(sd, ed)
        return [r.__dict__ for r in reviews], ""
    except Exception as exc:
        return [], str(exc)


@st.cache_data(ttl=900, show_spinner=False)
def _cached_google(start_str: str, end_str: str) -> Tuple[List[dict], str]:
    try:
        sd = datetime.strptime(start_str, "%Y-%m-%d").date()
        ed = datetime.strptime(end_str, "%Y-%m-%d").date()
        reviews = scrape_google_reviews(sd, ed)
        return [r.__dict__ for r in reviews], ""
    except Exception as exc:
        return [], str(exc)


@st.cache_data(ttl=900, show_spinner=False)
def _cached_app_stores(start_str: str, end_str: str) -> Tuple[List[dict], str]:
    try:
        sd = datetime.strptime(start_str, "%Y-%m-%d").date()
        ed = datetime.strptime(end_str, "%Y-%m-%d").date()
        reviews = scrape_app_store_reviews(sd, ed)
        return [r.__dict__ for r in reviews], ""
    except Exception as exc:
        return [], str(exc)


# ---------------------------------------------------------------------------
# Concurrent orchestrator
# ---------------------------------------------------------------------------

def _fetch_all_sources(
    sources: List[str],
    start_str: str,
    end_str: str,
    status_container=None,
) -> Tuple[Dict[str, Tuple[List[Review], str]], float]:
    """Run all selected scrapers concurrently and return per-source results + elapsed time."""

    dispatch: Dict[str, Callable] = {
        "trustpilot": lambda: _cached_trustpilot(start_str, end_str),
        "opinion_assurances": lambda: _cached_opinion_assurances(start_str, end_str),
        "google": lambda: _cached_google(start_str, end_str),
        "app_stores": lambda: _cached_app_stores(start_str, end_str),
    }

    results: Dict[str, Tuple[List[Review], str]] = {}
    t_start = time.time()

    # Friendly labels for progress display
    _DISPATCH_LABELS: Dict[str, str] = {
        "trustpilot": "⭐ Trustpilot",
        "opinion_assurances": "🛡️ Opinion Assurances",
        "google": "🔍 Google Reviews",
        "app_stores": "📱 App Stores (iOS + Android)",
    }

    # Show per-source progress
    source_status = {}
    if status_container:
        for src in sources:
            source_status[src] = status_container.status(
                f"{_DISPATCH_LABELS.get(src, src)} — scraping…",
                state="running",
            )

    with ThreadPoolExecutor(max_workers=len(sources)) as pool:
        future_to_source = {
            pool.submit(dispatch[src]): src
            for src in sources
            if src in dispatch
        }
        for future in as_completed(future_to_source):
            source = future_to_source[future]
            try:
                dicts, err = future.result()
                reviews = [Review(**d) for d in dicts]
                results[source] = (reviews, err)
                label = _DISPATCH_LABELS.get(source, source)
                if source in source_status:
                    if err:
                        source_status[source].update(
                            label=f"❌ {label} — Error",
                            state="error",
                        )
                    else:
                        source_status[source].update(
                            label=f"✅ {label} — {len(reviews)} reviews",
                            state="complete",
                        )
            except Exception as exc:
                results[source] = ([], str(exc))
                label = _DISPATCH_LABELS.get(source, source)
                if source in source_status:
                    source_status[source].update(
                        label=f"❌ {label} — Error",
                        state="error",
                    )

    elapsed = time.time() - t_start
    return results, elapsed


# ---------------------------------------------------------------------------
# Main area – Header
# ---------------------------------------------------------------------------

st.markdown(
    "<h1 style='margin-bottom: 0;'>📋 Alan Review Gatherer</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color: #6b7280; margin-top: 0.2rem; margin-bottom: 1.5rem;'>"
    "Collect and export Alan reviews from <b>Trustpilot</b>, <b>Opinion Assurances</b>, "
    "<b>Google Reviews</b>, <b>App Store</b> &amp; <b>Play Store</b> in one click.</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Fetch reviews
# ---------------------------------------------------------------------------
if st.button("🚀 Fetch Reviews", type="primary"):
    if start_date > end_date:
        st.error("Please fix the date range: start date must be before end date.")
    else:
        sources_selected: List[str] = []
        if use_trustpilot:
            sources_selected.append("trustpilot")
        if use_opinion_assurances:
            sources_selected.append("opinion_assurances")
        if use_google:
            sources_selected.append("google")
        if use_app_stores:
            sources_selected.append("app_stores")

        if not sources_selected:
            st.warning("Please select at least one source in the sidebar.")
        else:
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            status_container = st.container()

            results, elapsed = _fetch_all_sources(
                sources_selected,
                start_str,
                end_str,
                status_container=status_container,
            )

            # Aggregate
            all_reviews: List[Review] = []
            errors: Dict[str, str] = {}
            for src in sources_selected:
                reviews, err = results.get(src, ([], "Source not executed"))
                all_reviews.extend(reviews)
                if err:
                    errors[src] = err

            st.session_state["reviews"] = all_reviews
            st.session_state["errors"] = errors
            st.session_state["elapsed"] = elapsed
            st.session_state["sources_count"] = len(sources_selected)
            st.session_state["fetched_at"] = datetime.now().strftime("%H:%M:%S")

            st.toast(
                f"Done! {len(all_reviews)} reviews in {elapsed:.1f}s",
                icon="✅",
            )


# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if "reviews" in st.session_state and st.session_state["reviews"]:
    reviews_list: List[Review] = st.session_state["reviews"]
    df = reviews_to_dataframe(reviews_list)
    elapsed = st.session_state.get("elapsed", 0)
    fetched_at = st.session_state.get("fetched_at", "")
    errors = st.session_state.get("errors", {})

    # Map source keys to friendly names for display
    df["source_display"] = df["source"].map(_friendly_source)

    st.divider()

    # ---- Metrics row ----
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Reviews", f"{len(df):,}")
    with col2:
        avg_rating = df["rating"].mean() if not df.empty else 0
        st.metric("Avg Rating", f"{avg_rating:.1f} ⭐")
    with col3:
        sources_count = df["source"].nunique() if not df.empty else 0
        st.metric("Sources", sources_count)
    with col4:
        date_range_str = f"{start_date.strftime('%d/%m')} → {end_date.strftime('%d/%m/%Y')}"
        st.metric("Period", date_range_str)
    with col5:
        st.metric("Fetch Time", f"{elapsed:.1f}s", delta=f"at {fetched_at}" if fetched_at else None)

    # ---- Error summary (only in expander, not duplicated) ----
    if errors:
        with st.expander(f"⚠️ {len(errors)} source error(s)", expanded=False):
            for src, err in errors.items():
                st.warning(f"**{_friendly_source(src)}**: {err}")

    st.divider()

    # ---- Per-source breakdown ----
    st.subheader("📊 Reviews by Source")
    if not df.empty:
        source_summary = (
            df.groupby("source")
            .agg(count=("source", "size"), avg_rating=("rating", "mean"))
            .reset_index()
        )
        source_summary["avg_rating"] = source_summary["avg_rating"].round(2)

        source_cols = st.columns(len(source_summary))
        for i, (_, row) in enumerate(source_summary.iterrows()):
            src_key = row["source"]
            with source_cols[i]:
                st.metric(
                    label=f"{SOURCE_ICONS.get(src_key, '📡')} {_friendly_source(src_key)}",
                    value=f"{row['count']} reviews",
                    delta=f"Avg {row['avg_rating']} ⭐",
                )

    st.divider()

    # ---- Rating distribution chart ----
    st.subheader("📈 Rating Distribution")
    if not df.empty:
        rating_counts = (
            df["rating"]
            .apply(lambda x: int(round(x)))
            .value_counts()
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .rename("count")
        )
        chart_df = pd.DataFrame({
            "Stars": [f"{i}★" for i in rating_counts.index],
            "Reviews": rating_counts.values,
        })
        st.bar_chart(chart_df, x="Stars", y="Reviews", color="#4F46E5")

    st.divider()

    # ---- Source tabs + search + sort ----
    st.subheader("📝 All Reviews")

    # Filters row
    filter_col1, filter_col2, filter_col3 = st.columns([3, 1, 1])
    with filter_col1:
        search_query = st.text_input(
            "Search",
            placeholder="🔎 Filter by keyword in title or review text…",
            key="search_reviews",
            label_visibility="collapsed",
        )
    with filter_col2:
        sort_by = st.selectbox(
            "Sort",
            ["Date ↓ (newest)", "Date ↑ (oldest)", "Rating ↓ (best)", "Rating ↑ (worst)"],
            key="sort_reviews",
        )
    with filter_col3:
        min_rating = st.selectbox(
            "Min rating",
            [0, 1, 2, 3, 4, 5],
            format_func=lambda x: f"Min {x}★" if x else "All ratings",
            key="min_rating",
        )

    available_sources = sorted(df["source"].unique().tolist())
    tab_labels = [f"All ({len(df)})"] + [
        f"{SOURCE_ICONS.get(s, '📡')} {_friendly_source(s)} ({len(df[df['source'] == s])})"
        for s in available_sources
    ]
    tabs = st.tabs(tab_labels)

    def _apply_filters_and_sort(data: pd.DataFrame) -> pd.DataFrame:
        """Apply search, rating filter, and sort to a DataFrame."""
        filtered = data.copy()

        if search_query:
            q = search_query.lower()
            mask = (
                filtered["review_text"].str.lower().str.contains(q, na=False)
                | filtered["title"].str.lower().str.contains(q, na=False)
                | filtered["author"].str.lower().str.contains(q, na=False)
            )
            filtered = filtered[mask]

        if min_rating > 0:
            filtered = filtered[filtered["rating"] >= min_rating]

        if "newest" in sort_by:
            filtered = filtered.sort_values("date", ascending=False)
        elif "oldest" in sort_by:
            filtered = filtered.sort_values("date", ascending=True)
        elif "best" in sort_by:
            filtered = filtered.sort_values("rating", ascending=False)
        elif "worst" in sort_by:
            filtered = filtered.sort_values("rating", ascending=True)

        return filtered

    def _display_reviews_table(data: pd.DataFrame) -> None:
        """Render a filtered and sorted review table."""
        filtered = _apply_filters_and_sort(data)
        if filtered.empty:
            st.info("No reviews match the current filters.")
        else:
            st.caption(f"Showing **{len(filtered)}** of {len(data)} review(s)")
            # Use source_display instead of raw source key
            display_df = filtered[["source_display", "author", "rating", "date", "title", "review_text", "language"]].copy()
            column_config = {
                "source_display": st.column_config.TextColumn("Source", width="small"),
                "author": st.column_config.TextColumn("Author", width="small"),
                "rating": st.column_config.NumberColumn("Rating", format="%.0f ⭐", width="small"),
                "date": st.column_config.TextColumn("Date", width="small"),
                "title": st.column_config.TextColumn("Title", width="medium"),
                "review_text": st.column_config.TextColumn("Review", width="large"),
                "language": st.column_config.TextColumn("Lang", width="small"),
            }
            st.dataframe(
                display_df,
                hide_index=True,
                height=500,
                column_config=column_config,
            )

    with tabs[0]:
        _display_reviews_table(df)

    for idx, source_key in enumerate(available_sources):
        with tabs[idx + 1]:
            _display_reviews_table(df[df["source"] == source_key])

    st.divider()

    # ---- Download section ----
    st.subheader("⬇️ Export")
    st.caption(f"Export all **{len(df)}** reviews as CSV or JSON")

    csv_buffer = io.StringIO()
    df.drop(columns=["source_display"], errors="ignore").to_csv(csv_buffer, index=False, encoding="utf-8")
    csv_bytes = csv_buffer.getvalue().encode("utf-8-sig")  # BOM for Excel compat

    export_df = df.drop(columns=["source_display"], errors="ignore")
    json_bytes = export_df.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8")

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label=f"📄 Download CSV ({len(df)} reviews)",
            data=csv_bytes,
            file_name=f"alan_reviews_{start_date}_{end_date}.csv",
            mime="text/csv",
        )
    with col_dl2:
        st.download_button(
            label=f"📋 Download JSON ({len(df)} reviews)",
            data=json_bytes,
            file_name=f"alan_reviews_{start_date}_{end_date}.json",
            mime="application/json",
        )

elif "reviews" in st.session_state:
    # Reviews were fetched but none found
    st.divider()
    errors = st.session_state.get("errors", {})
    if errors:
        st.error("Some sources encountered errors:")
        for src, err in errors.items():
            st.warning(f"**{_friendly_source(src)}**: {err}")
    st.info("💡 No reviews found for the selected date range and sources. Try widening the date range.")

else:
    # Empty state – guide the user
    st.divider()
    st.markdown(
        """
        <div style="text-align: center; padding: 3rem 1rem; color: #9ca3af;">
            <p style="font-size: 3rem; margin-bottom: 0.5rem;">📋</p>
            <h3 style="color: #6b7280; margin-bottom: 0.5rem;">Ready to collect reviews</h3>
            <p style="font-size: 1.05rem; margin-bottom: 0.3rem;">
                1. Set your <b>date range</b> in the sidebar
            </p>
            <p style="font-size: 1.05rem; margin-bottom: 0.3rem;">
                2. Select the <b>sources</b> you want to scrape
            </p>
            <p style="font-size: 1.05rem;">
                3. Click <b style="color: #4F46E5;">🚀 Fetch Reviews</b> above
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

"""
Alan Monthly Review Report Generator
=====================================

Reads qualitative review data from an Excel file or from scraped reviews,
computes quantitative metrics, and uses OpenAI to generate a
comprehensive narrative Markdown report with charts.

Can be used as:
  - A library imported by the Streamlit app
  - A standalone CLI script:
      python generate_report.py --month 2026-02
      python generate_report.py --month 2026-02 --pdf

Requires:
    - OPENAI_API_KEY environment variable (or --api-key flag)
    - The Excel file "Alan qualitative reviews analysis 2025-2026.xlsx"
      in the same directory (or specify with --excel)
"""

from __future__ import annotations

import argparse
import calendar
import io
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd
import openai
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

EXCEL_FILE = "Alan qualitative reviews analysis 2025-2026.xlsx"

# Known platform names to filter out separator rows / headers
KNOWN_PLATFORMS = [
    "Google",
    "Trustpilot",
    "Play",
    "IOS",
    "Appfigures",
    "Opinion Assurances",
    "Opinion assurances",
]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def discover_sheets(excel_path: str) -> tuple[dict[str, str], list[str]]:
    """Auto-discover month sheets from the Excel file.

    Returns:
        month_map: e.g. {"2026-01": "January 2026 reviews", ...}
        month_order: sorted list of YYYY-MM keys
    """
    xls = pd.ExcelFile(excel_path)
    month_names_lower = {
        name.lower(): num for num, name in enumerate(calendar.month_name) if num
    }

    month_map: dict[str, str] = {}
    for sheet_name in xls.sheet_names:
        parts = sheet_name.strip().split()
        # Match pattern: "<MonthName> <YYYY> reviews"
        if len(parts) >= 3 and parts[-1].lower() == "reviews":
            month_str = parts[0].lower()
            year_str = parts[1]
            if month_str in month_names_lower and year_str.isdigit():
                month_num = month_names_lower[month_str]
                key = f"{year_str}-{month_num:02d}"
                month_map[key] = sheet_name

    month_order = sorted(month_map.keys())
    return month_map, month_order


def load_month_data(excel_path: str, sheet_name: str) -> pd.DataFrame:
    """Load and clean review data for a given month sheet."""
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # Keep only rows with a recognised platform (drops blanks, separators, etc.)
    df = df[df["Platform"].isin(KNOWN_PLATFORMS)].copy()

    # Coerce Rating to numeric
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    # Drop rows where Rating is NaN (shouldn't happen, but be safe)
    df = df.dropna(subset=["Rating"])

    return df.reset_index(drop=True)


def scraped_reviews_to_report_df(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """Convert scraped reviews DataFrame to the report format.

    Maps: source → Platform, rating → Rating, review_text → Content French,
    title → Topics (placeholder), etc.
    """
    SOURCE_TO_PLATFORM = {
        "google": "Google",
        "trustpilot": "Trustpilot",
        "opinion_assurances": "Opinion Assurances",
        "app_store_apple": "IOS",
        "app_store_google_play": "Play",
    }

    df = reviews_df.copy()
    df["Platform"] = df["source"].map(SOURCE_TO_PLATFORM).fillna(df["source"])
    df["Rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["Content French"] = df["review_text"].fillna("")
    df["Content English"] = ""  # Not available from scraping
    df["Topics"] = df.get("title", "").fillna("")
    df["Specifics"] = ""

    df = df.dropna(subset=["Rating"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _split_topics(series: pd.Series) -> list[str]:
    """Split comma-separated topic strings into a flat list."""
    topics: list[str] = []
    for value in series.dropna():
        for part in str(value).split(","):
            stripped = part.strip()
            if stripped:
                topics.append(stripped)
    return topics


def compute_metrics(df: pd.DataFrame, prev_df: Optional[pd.DataFrame] = None) -> dict:
    """Compute quantitative metrics from review data."""
    metrics: dict = {}

    # --- Basic counts ---
    metrics["total_reviews"] = len(df)
    metrics["avg_rating"] = round(df["Rating"].mean(), 2) if len(df) > 0 else 0
    metrics["five_star"] = int((df["Rating"] == 5).sum())
    metrics["four_star"] = int((df["Rating"] == 4).sum())
    metrics["three_star"] = int((df["Rating"] == 3).sum())
    metrics["two_star"] = int((df["Rating"] == 2).sum())
    metrics["one_star"] = int((df["Rating"] == 1).sum())

    # --- Delta vs previous month ---
    if prev_df is not None and len(prev_df) > 0:
        prev_avg = round(prev_df["Rating"].mean(), 2)
        metrics["avg_rating_delta"] = round(metrics["avg_rating"] - prev_avg, 2)
        metrics["prev_total"] = len(prev_df)
        metrics["prev_avg_rating"] = prev_avg

    # --- Rating distribution ---
    metrics["rating_distribution"] = (
        df["Rating"].value_counts().sort_index().to_dict()
    )

    # --- Platform breakdown ---
    metrics["platform_counts"] = df["Platform"].value_counts().to_dict()
    metrics["platform_ratings"] = (
        df.groupby("Platform")["Rating"]
        .agg(["mean", "count"])
        .round(2)
        .to_dict("index")
    )

    # --- Topic counts (all, positive, negative) ---
    if "Topics" in df.columns:
        metrics["topic_counts"] = dict(
            Counter(_split_topics(df["Topics"])).most_common()
        )
        metrics["positive_topic_counts"] = dict(
            Counter(_split_topics(df.loc[df["Rating"] >= 4, "Topics"])).most_common()
        )
        metrics["negative_topic_counts"] = dict(
            Counter(_split_topics(df.loc[df["Rating"] <= 2, "Topics"])).most_common()
        )
    else:
        metrics["topic_counts"] = {}
        metrics["positive_topic_counts"] = {}
        metrics["negative_topic_counts"] = {}

    return metrics


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

# Alan brand-ish palette
_COLORS = {
    "purple": "#6C5CE7",
    "blue": "#0984E3",
    "green": "#00B894",
    "red": "#D63031",
    "orange": "#E17055",
    "grey": "#B2BEC3",
    "light_purple": "#A29BFE",
    "light_blue": "#74B9FF",
    "dark": "#2D3436",
}

_RATING_COLORS = {
    1: _COLORS["red"],
    2: _COLORS["orange"],
    3: _COLORS["grey"],
    4: _COLORS["light_blue"],
    5: _COLORS["green"],
}


def _apply_style(ax: plt.Axes) -> None:
    """Apply a clean, modern look to a chart."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#DFE6E9")
    ax.spines["bottom"].set_color("#DFE6E9")
    ax.tick_params(colors=_COLORS["dark"], labelsize=9)
    ax.set_facecolor("white")


def generate_rating_distribution_chart(metrics: dict) -> plt.Figure:
    """Bar chart of the rating distribution. Returns the figure."""
    ratings = sorted(metrics["rating_distribution"].keys())
    counts = [metrics["rating_distribution"][r] for r in ratings]
    colors = [_RATING_COLORS.get(int(r), _COLORS["grey"]) for r in ratings]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(
        [f"{int(r)}★" for r in ratings], counts, color=colors, width=0.6,
        edgecolor="white", linewidth=1.2,
    )
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(count), ha="center", va="bottom", fontsize=11, fontweight="bold",
            color=_COLORS["dark"],
        )

    ax.set_ylabel("Number of reviews", fontsize=10, color=_COLORS["dark"])
    ax.set_title(
        "Rating Distribution", fontsize=13, fontweight="bold",
        color=_COLORS["dark"], pad=12,
    )
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _apply_style(ax)
    fig.tight_layout()
    return fig


def generate_platform_sentiment_chart(df: pd.DataFrame) -> plt.Figure:
    """Stacked horizontal bar chart: positive vs negative reviews per platform."""
    platform_col = df["Platform"].str.replace(
        "Opinion assurances", "Opinion Assurances"
    )
    positive = platform_col[df["Rating"] >= 4].value_counts()
    negative = platform_col[df["Rating"] <= 2].value_counts()

    platforms = sorted(set(positive.index) | set(negative.index))
    pos_vals = [positive.get(p, 0) for p in platforms]
    neg_vals = [negative.get(p, 0) for p in platforms]

    fig, ax = plt.subplots(
        figsize=(7, max(3, len(platforms) * 0.7 + 1))
    )
    y_pos = range(len(platforms))
    ax.barh(
        y_pos, pos_vals, color=_COLORS["green"], label="Positive (4-5★)",
        height=0.55, edgecolor="white", linewidth=1,
    )
    ax.barh(
        y_pos, [-n for n in neg_vals], color=_COLORS["red"],
        label="Negative (1-2★)", height=0.55, edgecolor="white", linewidth=1,
    )

    for i, (p, n) in enumerate(zip(pos_vals, neg_vals)):
        if p > 0:
            ax.text(
                p + 0.15, i, str(p), va="center", fontsize=9,
                color=_COLORS["green"], fontweight="bold",
            )
        if n > 0:
            ax.text(
                -n - 0.15, i, str(n), va="center", ha="right", fontsize=9,
                color=_COLORS["red"], fontweight="bold",
            )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(platforms, fontsize=10)
    ax.axvline(0, color="#DFE6E9", linewidth=1)
    ax.set_title(
        "Reviews by Platform (Positive vs Negative)", fontsize=13,
        fontweight="bold", color=_COLORS["dark"], pad=12,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    _apply_style(ax)
    fig.tight_layout()
    return fig


def generate_topics_chart(metrics: dict) -> Optional[plt.Figure]:
    """Horizontal bar chart of top positive and negative topics."""
    pos = metrics.get("positive_topic_counts", {})
    neg = metrics.get("negative_topic_counts", {})

    if not pos and not neg:
        return None

    pos_items = list(pos.items())[:6]
    neg_items = list(neg.items())[:6]

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(12, max(3.5, max(len(pos_items), len(neg_items), 1) * 0.55 + 1.5)),
    )

    if pos_items:
        labels, values = zip(*reversed(pos_items))
        ax1.barh(labels, values, color=_COLORS["green"], height=0.6, edgecolor="white")
        for i, v in enumerate(values):
            ax1.text(
                v + 0.1, i, str(v), va="center", fontsize=9,
                fontweight="bold", color=_COLORS["dark"],
            )
    ax1.set_title(
        "Top Positive Topics (4-5★)", fontsize=11, fontweight="bold",
        color=_COLORS["green"], pad=10,
    )
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _apply_style(ax1)

    if neg_items:
        labels, values = zip(*reversed(neg_items))
        ax2.barh(labels, values, color=_COLORS["red"], height=0.6, edgecolor="white")
        for i, v in enumerate(values):
            ax2.text(
                v + 0.1, i, str(v), va="center", fontsize=9,
                fontweight="bold", color=_COLORS["dark"],
            )
    ax2.set_title(
        "Top Negative Topics (1-2★)", fontsize=11, fontweight="bold",
        color=_COLORS["red"], pad=10,
    )
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _apply_style(ax2)

    fig.suptitle(
        "Topic Analysis", fontsize=13, fontweight="bold",
        color=_COLORS["dark"], y=1.02,
    )
    fig.tight_layout()
    return fig


def generate_month_comparison_chart(
    metrics: dict, prev_month_label: Optional[str], month_label: str,
) -> Optional[plt.Figure]:
    """Side-by-side bar chart comparing current vs previous month."""
    if "prev_avg_rating" not in metrics or prev_month_label is None:
        return None

    labels = [prev_month_label, month_label]
    avg_ratings = [metrics["prev_avg_rating"], metrics["avg_rating"]]
    totals = [metrics["prev_total"], metrics["total_reviews"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    bars1 = ax1.bar(
        labels, avg_ratings, color=[_COLORS["grey"], _COLORS["purple"]],
        width=0.5, edgecolor="white", linewidth=1.2,
    )
    for bar, val in zip(bars1, avg_ratings):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{val:.1f}", ha="center", va="bottom", fontsize=12,
            fontweight="bold", color=_COLORS["dark"],
        )
    ax1.set_ylim(0, 5.5)
    ax1.set_title("Avg Rating", fontsize=11, fontweight="bold", color=_COLORS["dark"])
    _apply_style(ax1)

    bars2 = ax2.bar(
        labels, totals, color=[_COLORS["grey"], _COLORS["blue"]],
        width=0.5, edgecolor="white", linewidth=1.2,
    )
    for bar, val in zip(bars2, totals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            str(val), ha="center", va="bottom", fontsize=12,
            fontweight="bold", color=_COLORS["dark"],
        )
    ax2.set_title(
        "Total Reviews", fontsize=11, fontweight="bold", color=_COLORS["dark"],
    )
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _apply_style(ax2)

    fig.suptitle(
        "Month-over-Month Comparison", fontsize=13, fontweight="bold",
        color=_COLORS["dark"], y=1.02,
    )
    fig.tight_layout()
    return fig


def generate_all_charts(
    df: pd.DataFrame,
    metrics: dict,
    month_label: str,
    prev_month_label: Optional[str] = None,
) -> dict[str, plt.Figure]:
    """Generate all charts and return a mapping of name → matplotlib figure."""
    charts: dict[str, plt.Figure] = {}

    charts["rating_distribution"] = generate_rating_distribution_chart(metrics)
    charts["platform_sentiment"] = generate_platform_sentiment_chart(df)

    topics_fig = generate_topics_chart(metrics)
    if topics_fig:
        charts["topics"] = topics_fig

    comparison_fig = generate_month_comparison_chart(
        metrics, prev_month_label, month_label,
    )
    if comparison_fig:
        charts["month_comparison"] = comparison_fig

    return charts


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _format_reviews_block(df: pd.DataFrame) -> str:
    """Render every review as structured text for the prompt."""
    lines: list[str] = []
    for idx, row in df.iterrows():
        lines.append(f"### Review {idx + 1}")
        lines.append(f"- Platform: {row.get('Platform', 'Unknown')}")
        lines.append(f"- Rating: {row.get('Rating', 'N/A')}")
        if "Topics" in row and pd.notna(row.get("Topics")) and str(row["Topics"]).strip():
            lines.append(f"- Topics: {row['Topics']}")
        if "Specifics" in row and pd.notna(row.get("Specifics")) and str(row["Specifics"]).strip():
            lines.append(f"- Specifics: {row['Specifics']}")
        eng = row.get("Content English", "")
        fra = row.get("Content French", "")
        if fra and pd.notna(fra) and str(fra).strip():
            lines.append(f"- Content (French): {fra}")
        if eng and pd.notna(eng) and str(eng).strip():
            lines.append(f"- Content (English): {eng}")
        lines.append("")
    return "\n".join(lines)


def _format_metrics_block(
    month_label: str, metrics: dict, prev_label: Optional[str],
) -> str:
    """Render computed metrics as readable text for the prompt."""
    parts: list[str] = [f"## Quantitative Metrics for {month_label}\n"]

    avg = metrics["avg_rating"]
    avg_str = f"- Average rating: {avg:.1f}"
    if "avg_rating_delta" in metrics and prev_label:
        delta = metrics["avg_rating_delta"]
        sign = "+" if delta >= 0 else ""
        avg_str += f" ({sign}{delta:.1f} vs {prev_label})"

    parts.append(f"- Total reviews: {metrics['total_reviews']}")
    parts.append(avg_str)
    parts.append(f"- 5-star reviews: {metrics['five_star']}")
    parts.append(f"- 4-star reviews: {metrics['four_star']}")
    parts.append(f"- 2-star reviews: {metrics['two_star']}")
    parts.append(f"- 1-star reviews: {metrics['one_star']}")

    parts.append("\n### Rating Distribution")
    for rating, count in sorted(metrics["rating_distribution"].items()):
        parts.append(f"- {rating:.0f} stars: {count} reviews")

    parts.append("\n### Platform Breakdown")
    for platform, data in metrics["platform_ratings"].items():
        parts.append(
            f"- {platform}: {data['count']:.0f} reviews, avg rating {data['mean']:.1f}"
        )

    if metrics.get("positive_topic_counts"):
        parts.append("\n### Top Positive Topics (4-5 star reviews)")
        for topic, count in metrics["positive_topic_counts"].items():
            parts.append(f"- {topic}: {count} mentions")

    if metrics.get("negative_topic_counts"):
        parts.append("\n### Top Negative Topics (1-2 star reviews)")
        for topic, count in metrics["negative_topic_counts"].items():
            parts.append(f"- {topic}: {count} mentions")

    return "\n".join(parts)


def build_prompt(
    month_label: str,
    metrics: dict,
    df: pd.DataFrame,
    prev_month_label: Optional[str] = None,
) -> tuple[str, str]:
    """Build the system + user prompt pair for the AI."""
    metrics_block = _format_metrics_block(month_label, metrics, prev_month_label)
    reviews_block = _format_reviews_block(df)

    system_prompt = (
        "You are a senior Voice-of-the-Customer analyst at Alan, a European "
        "health insurance company. You write extremely thorough, publication-ready "
        "monthly qualitative review analysis reports that are read by the CEO, "
        "VP Product, VP Customer Experience, and Head of Operations.\n\n"
        "Your reports are:\n"
        "- **Long and comprehensive** — every section should be deeply analysed "
        "with multiple paragraphs, not bullet-point summaries.\n"
        "- **Data-driven** — you cite precise mention counts, percentages, and "
        "rating distributions to support every claim.\n"
        "- **Rich in verbatim quotes** — you include many French quotes from the "
        "original reviews to illustrate points, with at least 2-3 quotes per theme.\n"
        "- **Analytical** — you identify root causes, correlations between themes, "
        "and explain *why* trends exist, not just *what* they are.\n"
        "- **Actionable** — every section ends with clear implications or "
        "recommendations for the relevant team.\n"
        "- **Contextual** — you compare with previous months when data is available "
        "and highlight emerging vs recurring issues.\n\n"
        "You write in English but all customer quotes remain in French."
    )

    user_prompt = f"""Write an extremely detailed and comprehensive monthly qualitative review analysis report for **{month_label}** based on the data below. The report should be VERY LONG and thorough — aim for at least 3000-4000 words. Every section must contain deep analysis with multiple paragraphs, verbatim French quotes, and actionable insights.

{metrics_block}

## All Reviews

{reviews_block}

---

Generate the report in **Markdown** with EXACTLY the following sections.
Use the quantitative data above and analyse EVERY individual review to write each section.
Be exhaustive — do not skip or summarise when you can elaborate.

# {month_label} - Alan Global Reviews ⭐

## ℹ️💡 Context & Methodology

Write a detailed context section (2-3 paragraphs):
- Explain that this is a qualitative analysis that complements quantitative dashboards.
- State the exact sample size, date range, and platforms covered.
- Note any methodological caveats (e.g. self-selection bias, platform skew).
- If previous month data is available, briefly frame the month-over-month context.
- Mention the types of reviewers (new members, long-term members, churned members) if discernible.

## 📊 Key Metrics

Present all key metrics in a clear, visually structured way using bold text and emojis:
- 🔄 Total reviews (with delta vs previous month if available)
- ⭐ Average rating (with delta vs previous month and interpretation)
- 🤩 Five-star reviews count and percentage of total
- ⭐ Four-star reviews count and percentage
- ⚠️ Two-star reviews count and percentage
- 💔 One-star reviews count and percentage
- 📊 Rating distribution analysis (is it bimodal? skewed? concentrated?)
- 📱 Platform breakdown (which platforms drive the most volume and which skew positive/negative)

Write a 1-2 paragraph interpretation of what the numbers tell us — don't just list them.

## 🔍 Executive Summary

Write a thorough executive summary (3-5 paragraphs):
- Overall sentiment assessment with precise numbers.
- The 2-3 most important positive signals and their business impact.
- The 2-3 most critical risk areas requiring immediate attention.
- Platform-specific trends and anomalies.
- How this month compares to the previous month (if data available).
- One-line recommendation for leadership.

## 🚨 Churn Risk Assessment

Provide a deep churn risk analysis (2-4 paragraphs):
- Identify ALL reviews where members explicitly mention wanting to leave, cancel, switch, or express strong dissatisfaction.
- Categorise churn signals: explicit intent to leave, implicit frustration suggesting churn, competitive mentions.
- Provide the exact count and include **at least 3-4** representative French quotes in blockquotes.
- Analyse the root causes driving churn intent.
- Estimate severity: how many are at immediate risk vs long-term risk?
- If no churn signals exist, analyse why and what protective factors are visible.
- End with specific retention recommendations.

## ⭐ What's Working — Detailed Positive Theme Analysis

For EACH of the top 5-6 positive themes, provide:
1. **Theme name** with exact mention count and percentage of positive reviews
2. **Deep explanation** (full paragraph): what specifically do reviewers praise? What makes Alan stand out?
3. **Multiple French quotes** (at least 2-3 per theme) in blockquotes showing the range of positive feedback
4. **Platform distribution**: which platforms mention this theme most?
5. **Business implication**: why does this matter and how can it be leveraged?

After listing all themes, write a **synthesis paragraph** connecting the positive themes and explaining what they reveal about Alan's core value proposition.

## 💔 What's Not Working — Detailed Negative Theme Analysis

For EACH of the top 5-6 negative themes, provide:
1. **Theme name** with exact mention count and percentage of negative reviews
2. **Root cause analysis** (full paragraph): what is the underlying problem? Is it a product issue, process issue, or communication issue?
3. **Impact assessment**: how does this issue affect member satisfaction and retention?
4. **Multiple French quotes** (at least 2-3 per theme) in blockquotes showing the range and severity of complaints
5. **Platform distribution**: where is the complaint concentrated?
6. **Specific recommendation**: what should the relevant team do to address this?

After listing all themes, write a **synthesis paragraph** connecting the negative themes and identifying systemic issues or patterns.

## 🔄 Sentiment Trends & Patterns

Analyse cross-cutting patterns (2-3 paragraphs):
- Are there correlations between themes (e.g. people who complain about X also mention Y)?
- Are certain issues platform-specific or universal?
- Are there demographic or usage patterns visible (new members vs long-term)?
- How has the sentiment mix shifted compared to the previous month (if data available)?
- Identify any emerging issues that appeared this month for the first time.

## 💡 Product & Technical Signals

Provide an exhaustive list of actionable product feedback, grouped by category:

### 🐛 Bugs & Technical Issues
- List each specific bug or technical problem mentioned, with French quotes.
- Assess severity and frequency.

### 🆕 Feature Requests
- List each feature request, with French quotes.
- Assess feasibility and potential impact.

### 🎨 UX & Design Improvements
- List UX friction points and improvement suggestions.
- Include specific user journey pain points.

### 📞 Customer Service & Operations
- List operational issues (response time, process friction, claim handling).
- Include specific French quotes about service interactions.

If a category has no signals, state that explicitly.

## 📱 Reviews by Platform — Detailed Breakdown

For EACH platform (Google, Trustpilot, App Store, Play Store, Opinion Assurances, etc.):
- Number of reviews and average rating.
- Dominant sentiment (positive, negative, mixed).
- Key themes specific to this platform.
- 1-2 representative French quotes.
- Why this platform may skew differently (audience, timing, review triggers).

Write a **comparison paragraph** at the end highlighting the most striking cross-platform differences.

## 🏆 Notable Reviews — Highlights & Lowlights

Select 3-4 of the most impactful reviews (positive AND negative) and provide:
- The full or near-full French quote in a blockquote.
- Platform and rating.
- Why this review matters (reputation risk, product insight, competitive signal).
- Recommended action if applicable.

## 📋 Recommendations & Action Items

Provide a prioritised list of 5-8 concrete recommendations:
1. **Immediate actions** (this week): urgent fixes or responses needed.
2. **Short-term actions** (this month): improvements to plan and execute.
3. **Strategic actions** (this quarter): larger initiatives informed by the data.

Each recommendation should reference the specific data/theme that drives it.

---

CRITICAL GUIDELINES — FOLLOW ALL OF THESE:
- This report must be **extremely detailed and long** — at least 3000-4000 words. Do NOT be brief.
- All quotes MUST be in FRENCH (from the original "Content French" field). Include MANY quotes.
- Use precise mention counts and percentages — be exact, not vague.
- Be deeply analytical and insightful — explain WHY, not just WHAT.
- Flag the most reputationally damaging reviews with urgency.
- Keep the tone professional, strategic, and direct.
- If data is limited, acknowledge it but still extract maximum insight.
- Do NOT include any image references or chart placeholders — charts will be inserted automatically.
- Write full paragraphs, not just bullet lists — this is a narrative report, not a dashboard.
- Every section should feel like it was written by a senior analyst who deeply understands the business."""

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# AI report generation
# ---------------------------------------------------------------------------

def generate_report_text(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    model: str = "gpt-5.2",
) -> str:
    """Call OpenAI API to generate the Markdown report."""
    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=16000,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Full pipeline (for library usage)
# ---------------------------------------------------------------------------

def generate_report_from_df(
    df: pd.DataFrame,
    month_label: str,
    api_key: str,
    prev_df: Optional[pd.DataFrame] = None,
    prev_month_label: Optional[str] = None,
    model: str = "gpt-5.2",
) -> tuple[str, dict, dict[str, plt.Figure]]:
    """Full pipeline: metrics + charts + AI report from a DataFrame.

    Args:
        df: Review data with columns Platform, Rating, Content French, etc.
        month_label: Human-readable month label (e.g. "February '26").
        api_key: OpenAI API key.
        prev_df: Previous month data for comparison (optional).
        prev_month_label: Human-readable previous month label (optional).
        model: OpenAI model to use.

    Returns:
        (report_markdown, metrics_dict, charts_dict)
    """
    metrics = compute_metrics(df, prev_df)
    charts = generate_all_charts(df, metrics, month_label, prev_month_label)
    system_prompt, user_prompt = build_prompt(
        month_label, metrics, df, prev_month_label,
    )
    report_md = generate_report_text(system_prompt, user_prompt, api_key, model)

    return report_md, metrics, charts


# ---------------------------------------------------------------------------
# PDF export (reusable)
# ---------------------------------------------------------------------------

_PDF_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  @page { size: A4; margin: 2cm 2.2cm; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Helvetica, Arial, sans-serif;
    color: #2D3436; font-size: 11pt; line-height: 1.6;
  }
  h1 { font-size: 22pt; color: #6C5CE7; border-bottom: 3px solid #6C5CE7;
       padding-bottom: 6px; margin-top: 0; }
  h2 { font-size: 15pt; color: #2D3436; margin-top: 28px;
       border-bottom: 1px solid #DFE6E9; padding-bottom: 4px; }
  h3 { font-size: 12pt; color: #636E72; }
  blockquote { border-left: 4px solid #6C5CE7; margin: 10px 0;
               padding: 6px 14px; background: #F8F9FA; color: #636E72;
               font-style: italic; }
  strong { color: #2D3436; }
  ul, ol { padding-left: 22px; }
  li { margin-bottom: 4px; }
  table { border-collapse: collapse; width: 100%; margin: 12px 0; }
  th, td { border: 1px solid #DFE6E9; padding: 6px 10px; font-size: 10pt; }
  th { background: #F8F9FA; font-weight: 600; }
</style>
</head>
<body>
{body}
</body>
</html>
"""


def convert_md_to_pdf_bytes(md_text: str) -> bytes:
    """Convert Markdown text to a styled PDF and return as bytes.

    This is designed for in-memory usage (e.g. Streamlit download button).
    """
    import markdown as md_lib
    from weasyprint import HTML

    html_body = md_lib.markdown(
        md_text, extensions=["extra", "sane_lists", "tables"],
    )
    full_html = _PDF_HTML_TEMPLATE.format(body=html_body)
    return HTML(string=full_html).write_pdf()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an Alan monthly qualitative review report.",
    )
    parser.add_argument(
        "--month", required=True,
        help="Target month in YYYY-MM format (e.g. 2026-02).",
    )
    parser.add_argument(
        "--excel", default=None,
        help=f"Path to the Excel file (default: '{EXCEL_FILE}' in script directory).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file path (default: report_YYYY-MM.md in script directory).",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="OpenAI API key (default: OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--pdf", action="store_true",
        help="Also export the report as a styled PDF (requires weasyprint).",
    )
    args = parser.parse_args()

    # -- Load .env file from script directory --
    try:
        from dotenv import load_dotenv
        script_dir = Path(__file__).resolve().parent
        load_dotenv(script_dir / ".env")
    except ImportError:
        script_dir = Path(__file__).resolve().parent

    # -- Resolve paths --
    excel_path = Path(args.excel) if args.excel else script_dir / EXCEL_FILE
    output_path = (
        Path(args.output) if args.output else script_dir / f"report_{args.month}.md"
    )

    # -- API key --
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: Provide an OpenAI API key via OPENAI_API_KEY env var "
            "or --api-key flag."
        )
        sys.exit(1)

    # -- Validate Excel --
    if not excel_path.exists():
        print(f"Error: Excel file not found at {excel_path}")
        sys.exit(1)

    # -- Discover sheets --
    month_map, month_order = discover_sheets(str(excel_path))
    if args.month not in month_map:
        print(f"Error: No data sheet found for month '{args.month}'.")
        print(f"Available months: {', '.join(month_order)}")
        sys.exit(1)

    # -- Load target month --
    target_sheet = month_map[args.month]
    print(f"📂 Loading data from sheet: {target_sheet}")
    df = load_month_data(str(excel_path), target_sheet)
    print(f"   ➜ {len(df)} reviews loaded")

    # -- Load previous month for delta comparison --
    prev_df = None
    prev_month_label = None
    month_idx = month_order.index(args.month)
    if month_idx > 0:
        prev_key = month_order[month_idx - 1]
        prev_sheet = month_map[prev_key]
        print(f"📂 Loading previous month: {prev_sheet}")
        prev_df = load_month_data(str(excel_path), prev_sheet)
        y, m = prev_key.split("-")
        prev_month_label = f"{calendar.month_name[int(m)]} {y}"
        print(f"   ➜ {len(prev_df)} reviews for comparison")

    # -- Build month label --
    y, m = args.month.split("-")
    month_label = f"{calendar.month_name[int(m)]} '{y[2:]}"

    # -- Run full pipeline --
    print("📊 Computing metrics & generating charts…")
    print("🤖 Building prompt and calling OpenAI…")
    report_md, metrics, charts = generate_report_from_df(
        df, month_label, api_key, prev_df, prev_month_label,
    )

    # -- Save charts to disk --
    charts_dir = output_path.parent / f"charts_{args.month}"
    charts_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in charts.items():
        fig.savefig(
            charts_dir / f"{name}.png", dpi=150, bbox_inches="tight", facecolor="white",
        )
        plt.close(fig)
    print(f"   ➜ {len(charts)} charts saved to {charts_dir}/")

    # -- Write Markdown output --
    output_path.write_text(report_md, encoding="utf-8")
    print(f"\n✅ Markdown report written to: {output_path}")
    print(f"   {len(report_md):,} characters")

    # -- Export to PDF if requested --
    if args.pdf:
        try:
            import markdown as md_lib
            from weasyprint import HTML

            pdf_path = output_path.with_suffix(".pdf")
            print("📄 Exporting to PDF…")
            html_body = md_lib.markdown(
                report_md, extensions=["extra", "sane_lists"],
            )
            _HTML_TEMPLATE = """\
<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
body {{ font-family: sans-serif; color: #2D3436; font-size: 11pt; line-height: 1.55; }}
h1 {{ color: #6C5CE7; border-bottom: 3px solid #6C5CE7; padding-bottom: 6px; }}
h2 {{ border-bottom: 1px solid #DFE6E9; padding-bottom: 4px; }}
blockquote {{ border-left: 4px solid #6C5CE7; padding: 6px 14px; background: #F8F9FA; font-style: italic; }}
</style></head><body>{body}</body></html>"""
            full_html = _HTML_TEMPLATE.format(body=html_body)
            HTML(
                string=full_html, base_url=str(output_path.parent),
            ).write_pdf(str(pdf_path))
            print(f"✅ PDF report written to: {pdf_path}")
        except ImportError:
            print("⚠️ PDF export requires 'markdown' and 'weasyprint' packages.")


if __name__ == "__main__":
    main()

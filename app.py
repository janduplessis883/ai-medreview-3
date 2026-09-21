"""Friends and Family Test review dashboard for PCNs and surgeries."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib

import altair as alt
import pandas as pd
import streamlit as st


DATA_DIR = Path(__file__).parent / "data"
SAMPLE_DATA_PATH = DATA_DIR / "sample_jev_analyzed_NEW.csv"
NEW_DATA_PATH = DATA_DIR / "new_data_jev_analyzed.csv"
CONFIG_PATH = Path(__file__).parent / ".streamlit" / "config.toml"
HIGH_URGENCY = 2.5
LOW_CONFIDENCE = 0.60


@st.cache_data
def load_reviews(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize the original test-output schema and the new batch-output schema.
    if "time" not in df.columns and "date" in df.columns:
        df["time"] = df["date"]
    if "free_text" not in df.columns and "review" in df.columns:
        df["free_text"] = df["review"]
    required = {"time", "free_text", "pcn", "surgery"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Analyzed CSV is missing required columns: {', '.join(sorted(missing))}")
    df["review_date"] = pd.to_datetime(df["time"], errors="coerce")
    numeric = [
        "rating",
        "jev_sentiment_confidence",
        "jev_primary_topic_confidence",
        "jev_sentiment_strength",
        "jev_actionability",
        "jev_urgency",
    ]
    for column in numeric:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["review_id"] = range(1, len(df) + 1)
    df["needs_review"] = (
        (df["jev_sentiment_confidence"] < 0.65)
        | (df["jev_primary_topic_confidence"] < LOW_CONFIDENCE)
        | (df["jev_urgency"] >= HIGH_URGENCY)
        | df["jev_primary_topic"].eq("Respect, Dignity, Privacy and Inclusion")
    )
    return df


def pct(value: float) -> str:
    return f"{value:.0%}"


@st.cache_data
def load_chart_colors(path: str) -> list[str]:
    """Read the categorical chart palette from Streamlit's project config."""
    config_path = Path(path)
    if not config_path.exists():
        return ["#FF9900", "#232F3E", "#007185", "#067D62", "#FFD814", "#B12704", "#565959"]
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)
    colors = config.get("theme", {}).get("chartCategoricalColors", [])
    return colors or ["#FF9900", "#232F3E", "#007185", "#067D62", "#FFD814", "#B12704", "#565959"]


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("Review filters")
        pcn = st.multiselect("PCN", sorted(df["pcn"].dropna().unique()), default=[])
        surgery = st.multiselect("Surgery", sorted(df["surgery"].dropna().unique()), default=[])
        sentiment = st.multiselect(
            "JEV sentiment", sorted(df["jev_sentiment"].dropna().unique()), default=[]
        )
        topics = st.multiselect(
            "Primary topic", sorted(df["jev_primary_topic"].dropna().unique()), default=[]
        )
        min_actionability = st.slider(
            "Minimum actionability",
            min_value=0.0,
            max_value=4.0,
            value=0.0,
            step=0.1,
            help="Show reviews with a JEV actionability score at or above this value.",
        )
        min_urgency = st.slider(
            "Minimum urgency",
            min_value=0.0,
            max_value=4.0,
            value=0.0,
            step=0.1,
            help="Show reviews with a JEV urgency score at or above this value.",
        )
        date_min = df["review_date"].min().date()
        date_max = df["review_date"].max().date()
        dates = st.date_input("Review date", value=(date_min, date_max), min_value=date_min, max_value=date_max)
        review_queue = st.checkbox("Only show reviews needing human review")

        st.divider()
        st.caption("JEV routing thresholds")
        st.caption(f"Low confidence: < {LOW_CONFIDENCE:.0%}")
        st.caption(f"High urgency: ≥ {HIGH_URGENCY:.1f}")

    filtered = df.copy()
    if pcn:
        filtered = filtered[filtered["pcn"].isin(pcn)]
    if surgery:
        filtered = filtered[filtered["surgery"].isin(surgery)]
    if sentiment:
        filtered = filtered[filtered["jev_sentiment"].isin(sentiment)]
    if topics:
        filtered = filtered[filtered["jev_primary_topic"].isin(topics)]
    filtered = filtered[filtered["jev_actionability"] >= min_actionability]
    filtered = filtered[filtered["jev_urgency"] >= min_urgency]
    if isinstance(dates, tuple) and len(dates) == 2:
        filtered = filtered[filtered["review_date"].dt.date.between(dates[0], dates[1])]
    if review_queue:
        filtered = filtered[filtered["needs_review"]]
    return filtered


def render_kpis(df: pd.DataFrame) -> None:
    negative = df["jev_sentiment"].eq("Negative").mean() if len(df) else 0
    actionable = (df["jev_actionability"] >= 2).mean() if len(df) else 0
    reviewed = df["needs_review"].mean() if len(df) else 0
    with st.container(horizontal=True):
        st.metric("Reviews", f"{len(df):,}", border=True)
        st.metric("Positive sentiment", pct(df["jev_sentiment"].eq("Positive").mean()) if len(df) else "0%", border=True)
        st.metric("Negative sentiment", pct(negative), border=True)
        st.metric("Actionable feedback", pct(actionable), border=True)
        st.metric("Human review queue", f"{int(df['needs_review'].sum()):,} ({pct(reviewed)})", border=True)


def render_overview(df: pd.DataFrame, chart_colors: list[str]) -> None:
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("Sentiment mix")
            sentiment = df["jev_sentiment"].value_counts().rename_axis("sentiment").reset_index(name="reviews")
            sentiment_chart = alt.Chart(sentiment).mark_bar().encode(
                x=alt.X("sentiment:N", title=None),
                y=alt.Y("reviews:Q", title="Reviews"),
                color=alt.Color("sentiment:N", title="Sentiment", scale=alt.Scale(range=chart_colors)),
                tooltip=["sentiment:N", "reviews:Q"],
            )
            st.altair_chart(sentiment_chart)
    with right:
        with st.container(border=True):
            st.subheader("Primary topics")
            topics = df["jev_primary_topic"].value_counts().head(10).sort_values()
            topic_data = topics.rename("reviews").reset_index().rename(columns={"jev_primary_topic": "topic"})
            topic_chart = alt.Chart(topic_data).mark_bar().encode(
                x=alt.X("reviews:Q", title="Reviews"),
                y=alt.Y("topic:N", sort="-x", title=None),
                color=alt.Color("topic:N", title="Topic", scale=alt.Scale(range=chart_colors)),
                tooltip=["topic:N", "reviews:Q"],
            )
            st.altair_chart(topic_chart)

    st.subheader("Reviews received per month")
    monthly = (
        df.assign(month=df["review_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)
        .size()
        .rename(columns={"size": "reviews"})
    )
    monthly_chart = alt.Chart(monthly).mark_bar(color=chart_colors[2]).encode(
        x=alt.X("month:T", title="Month"),
        y=alt.Y("reviews:Q", title="Reviews"),
        tooltip=[alt.Tooltip("month:T", title="Month", format="%b %Y"), "reviews:Q"],
    )
    st.altair_chart(monthly_chart)

    st.subheader("PCN comparison")
    pcn_summary = (
        df.groupby("pcn", dropna=False)
        .agg(
            reviews=("review_id", "count"),
            positive=("jev_sentiment", lambda s: s.eq("Positive").mean()),
            negative=("jev_sentiment", lambda s: s.eq("Negative").mean()),
            actionable=("jev_actionability", lambda s: (s >= 2).mean()),
            review_queue=("needs_review", "sum"),
        )
        .reset_index()
        .sort_values("reviews", ascending=False)
    )
    st.dataframe(
        pcn_summary,
        hide_index=True,
        column_config={
            "pcn": st.column_config.TextColumn("PCN"),
            "positive": st.column_config.NumberColumn("Positive", format="%.0f%%"),
            "negative": st.column_config.NumberColumn("Negative", format="%.0f%%"),
            "actionable": st.column_config.NumberColumn("Actionable", format="%.0f%%"),
            "review_queue": st.column_config.NumberColumn("Review queue"),
        },
    )


def render_surgery_view(df: pd.DataFrame, chart_colors: list[str]) -> None:
    st.subheader("Surgery comparison")
    surgery_summary = (
        df.groupby(["pcn", "surgery"], dropna=False)
        .agg(
            reviews=("review_id", "count"),
            positive=("jev_sentiment", lambda s: s.eq("Positive").mean()),
            negative=("jev_sentiment", lambda s: s.eq("Negative").mean()),
            avg_actionability=("jev_actionability", "mean"),
            review_queue=("needs_review", "sum"),
        )
        .reset_index()
        .sort_values(["pcn", "reviews"], ascending=[True, False])
    )
    st.dataframe(
        surgery_summary,
        hide_index=True,
        column_config={
            "pcn": "PCN",
            "surgery": "Surgery",
            "positive": st.column_config.NumberColumn("Positive", format="%.0f%%"),
            "negative": st.column_config.NumberColumn("Negative", format="%.0f%%"),
            "avg_actionability": st.column_config.NumberColumn("Avg actionability", format="%.2f"),
        },
    )

    trend = df.assign(month=df["review_date"].dt.to_period("M").astype(str)).groupby("month").size().reset_index(name="reviews")
    trend_chart = alt.Chart(trend).mark_line(point=True).encode(
        x=alt.X("month:N", title="Month"),
        y=alt.Y("reviews:Q", title="Reviews"),
        color=alt.value(chart_colors[0]),
        tooltip=["month:N", "reviews:Q"],
    )
    st.altair_chart(trend_chart)


def render_review_queue(df: pd.DataFrame) -> None:
    st.subheader("Review queue")
    queue = df[df["needs_review"]].copy().sort_values(["jev_urgency", "jev_actionability"], ascending=False)
    if queue.empty:
        st.success("No reviews in the human review queue for the current filters.")
        return
    display = queue[[
        "review_id", "review_date", "pcn", "surgery", "jev_sentiment", "jev_primary_topic",
        "jev_primary_topic_confidence", "jev_actionability", "jev_urgency", "free_text",
    ]].rename(columns={
        "review_id": "ID", "review_date": "Date", "pcn": "PCN", "surgery": "Surgery",
        "jev_sentiment": "Sentiment", "jev_primary_topic": "Primary topic",
        "jev_primary_topic_confidence": "Topic confidence", "jev_actionability": "Actionability",
        "jev_urgency": "Urgency", "free_text": "Review",
    })
    st.dataframe(display, hide_index=True, height=430, column_config={
        "Date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
        "Topic confidence": st.column_config.ProgressColumn("Topic confidence", min_value=0, max_value=1, format="%.0%%"),
        "Review": st.column_config.TextColumn("Review", width="large"),
    })


def render_review_explorer(df: pd.DataFrame) -> None:
    st.subheader("Review explorer")
    if df.empty:
        st.info("No reviews match the current filters.")
        return
    options = df["review_id"].tolist()
    selected = st.selectbox("Select a review", options, format_func=lambda value: f"Review {value}")
    row = df.loc[df["review_id"].eq(selected)].iloc[0]
    with st.container(border=True):
        st.write(row["free_text"])
        cols = st.columns(4)
        cols[0].metric("Sentiment", row["jev_sentiment"])
        primary_topic = str(row["jev_primary_topic"])
        compact_topic = primary_topic if len(primary_topic) <= 18 else f"{primary_topic[:17]}…"
        cols[1].metric("Primary topic", compact_topic, help=primary_topic)
        cols[2].metric("Actionability", f"{row['jev_actionability']:.2f}")
        cols[3].metric("Urgency", f"{row['jev_urgency']:.2f}")
        st.caption(f"{row['pcn']} · {row['surgery']} · {row['review_date']:%d %b %Y}")
        secondary = row.get("jev_secondary_topics", "")
        st.markdown("### Primary category")
        st.write(row["jev_primary_topic"])
        st.markdown("### Secondary category")
        st.write(secondary or "None recorded")
        with st.expander("JEV confidence and raw topic probabilities"):
            st.json({
                "sentiment_confidence": row["jev_sentiment_confidence"],
                "primary_topic_confidence": row["jev_primary_topic_confidence"],
                "sentiment_strength": row["jev_sentiment_strength"],
                "actionability_confidence": row["jev_actionability_confidence"],
                "urgency_confidence": row["jev_urgency_confidence"],
                "secondary_topic_scores": json.loads(row["jev_secondary_topic_scores"]),
            })
            if "jev_raw_output_json" in row.index and row["jev_raw_output_json"]:
                st.markdown("**Full JEV output**")
                try:
                    st.json(json.loads(row["jev_raw_output_json"]))
                except (TypeError, json.JSONDecodeError):
                    st.code(str(row["jev_raw_output_json"]), language="json")


def main() -> None:
    st.set_page_config(page_title="AI MedReview", page_icon=":material/medical_services:", layout="wide")
    st.title("AI MedReview")
    st.caption("Friends and Family Test intelligence for PCN and surgery teams")
    data_path = NEW_DATA_PATH if NEW_DATA_PATH.exists() else SAMPLE_DATA_PATH
    if not data_path.exists():
        st.error(f"Could not find an analyzed CSV at {NEW_DATA_PATH} or {SAMPLE_DATA_PATH}")
        st.stop()
    try:
        df = load_reviews(str(data_path))
    except ValueError as error:
        st.error(str(error))
        st.stop()
    chart_colors = load_chart_colors(str(CONFIG_PATH))
    st.caption(f"Data source: `{data_path.name}`")
    filtered = apply_filters(df)
    st.caption(f"Showing {len(filtered):,} of {len(df):,} analyzed reviews")
    render_kpis(filtered)
    overview, surgeries, queue, explorer = st.tabs(["PCN overview", "Surgery comparison", "Review queue", "Review explorer"])
    with overview:
        render_overview(filtered, chart_colors)
    with surgeries:
        render_surgery_view(filtered, chart_colors)
    with queue:
        render_review_queue(filtered)
    with explorer:
        render_review_explorer(filtered)
    st.divider()
    st.caption("JEV provides typed judgments; the review queue is derived from explicit confidence, urgency, and inclusion rules.")


if __name__ == "__main__":
    main()

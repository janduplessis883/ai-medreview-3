import json
from pathlib import Path
import tomllib

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_PATH = DATA_DIR / "sample_jev_analyzed_NEW.csv"
NEW_PATH = DATA_DIR / "new_data_jev_analyzed.csv"
CONFIG_PATH = BASE_DIR / ".streamlit" / "config.toml"


def data_path() -> Path:
    return NEW_PATH if NEW_PATH.exists() else SAMPLE_PATH


def load_reviews() -> pd.DataFrame:
    df = pd.read_csv(data_path()).fillna("")
    if "time" not in df and "date" in df:
        df["time"] = df["date"]
    if "free_text" not in df and "review" in df:
        df["free_text"] = df["review"]
    required = {"time", "free_text", "pcn", "surgery"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    df["review_date"] = pd.to_datetime(df["time"], errors="coerce")
    for column in ["jev_sentiment_confidence", "jev_primary_topic_confidence", "jev_actionability", "jev_urgency", "jev_safety_or_inclusion_concern"]:
        if column not in df:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    df["review_id"] = range(1, len(df) + 1)
    df["needs_review"] = (
        (df["jev_sentiment_confidence"] < 0.65)
        | (df["jev_primary_topic_confidence"] < 0.60)
        | (df["jev_urgency"] >= 2.5)
        | (df["jev_safety_or_inclusion_concern"] >= 0.7)
        | df.get("jev_primary_topic", "").eq("Respect, Dignity, Privacy and Inclusion")
    )
    return df


def filter_reviews(df: pd.DataFrame, params) -> pd.DataFrame:
    result = df.copy()
    for column in ["pcn", "surgery", "jev_sentiment", "jev_primary_topic"]:
        values = params.getlist(column)
        if values:
            result = result[result[column].isin(values)]
    result = result[result["jev_actionability"] >= float(params.get("min_actionability", 0))]
    result = result[result["jev_urgency"] >= float(params.get("min_urgency", 0))]
    result = result[result["jev_safety_or_inclusion_concern"] >= float(params.get("min_safety_concern", 0))]
    start, end = params.get("start"), params.get("end")
    if start:
        result = result[result["review_date"].dt.date >= pd.to_datetime(start).date()]
    if end:
        result = result[result["review_date"].dt.date <= pd.to_datetime(end).date()]
    if params.get("review_queue") == "1":
        result = result[result["needs_review"]]
    return result


def chart_colors() -> list[str]:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("rb") as handle:
            colors = tomllib.load(handle).get("theme", {}).get("chartCategoricalColors", [])
            if colors:
                return colors
    return ["#FF9900", "#232F3E", "#007185", "#067D62", "#FFD814", "#B12704", "#565959"]


def jsonable_review(row) -> dict:
    payload = row.to_dict()
    for key, value in payload.items():
        if isinstance(value, pd.Timestamp):
            payload[key] = value.isoformat()
    if payload.get("jev_raw_output_json"):
        try:
            payload["jev_raw_output_json"] = json.loads(payload["jev_raw_output_json"])
        except json.JSONDecodeError:
            pass
    return payload

#!/usr/bin/env python3
"""Analyze data/new_data.csv with TypeSafe Jev and write an enriched CSV."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy
from tqdm import tqdm

from src.ai_medreview_3.analyze_reviews_jev import (
    DEFAULT_MODEL,
    answer_value,
    build_questions,
    parse_choice_answer,
)


DEFAULT_INPUT = Path(__file__).parent / "data" / "new_data.csv"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "new_data_jev_analyzed.csv"
CHECKPOINT_EVERY = 100


def to_jsonable(value: Any) -> Any:
    """Convert SDK response objects into JSON-safe nested values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return to_jsonable(value.dict())
    if hasattr(value, "__dict__"):
        return to_jsonable(vars(value))
    return str(value)


def serialize_response(response: Any) -> str:
    """Persist the complete SDK response, including answer distributions."""
    return json.dumps(to_jsonable(response), ensure_ascii=False, sort_keys=True, default=str)


async def analyze_one(
    client: AsyncTypeSafeClient,
    text: str,
    *,
    questions: dict[str, Any],
    secondary_topics: list[str],
    timeout: float,
    secondary_threshold: float,
) -> dict[str, Any]:
    response = await client.system_one(state={"review": text}, questions=questions, timeout=timeout)
    sentiment_answer = response.choices["sentiment"]
    primary_answer = response.choices["primary_topic"]
    primary_topic, primary_confidence = parse_choice_answer(response, "primary_topic")
    secondary_scores = {
        topic: answer_value(response.nouls[f"secondary_{index}"], "noul")
        for index, topic in enumerate(secondary_topics)
    }
    selected_secondary = [
        topic
        for topic, probability in secondary_scores.items()
        if topic != primary_topic and probability >= secondary_threshold
    ]
    return {
        "jev_sentiment": answer_value(sentiment_answer, "choice"),
        "jev_sentiment_confidence": answer_value(sentiment_answer, "confidence"),
        "jev_primary_topic": primary_topic,
        "jev_primary_topic_confidence": primary_confidence,
        "jev_secondary_topics": " | ".join(selected_secondary),
        "jev_secondary_topic_scores": json.dumps(secondary_scores, sort_keys=True),
        "jev_sentiment_strength": answer_value(response.scores["sentiment_strength"], "score"),
        "jev_sentiment_strength_confidence": answer_value(response.scores["sentiment_strength"], "confidence"),
        "jev_actionability": answer_value(response.scores["actionability"], "score"),
        "jev_actionability_confidence": answer_value(response.scores["actionability"], "confidence"),
        "jev_urgency": answer_value(response.scores["urgency"], "score"),
        "jev_urgency_confidence": answer_value(response.scores["urgency"], "confidence"),
        "jev_safety_or_inclusion_concern": answer_value(
            response.nouls["safety_or_inclusion_concern"], "noul"
        ),
        "jev_raw_output_json": serialize_response(response),
    }


async def analyze_file(
    input_path: Path,
    output_path: Path,
    *,
    api_key: str,
    model: str,
    timeout: float,
    max_retries: int,
    concurrency: int,
    secondary_threshold: float,
) -> None:
    reviews = pd.read_csv(input_path).fillna("")
    text_column = "free_text" if "free_text" in reviews.columns else "review"
    if text_column not in reviews.columns:
        raise ValueError("Input CSV must contain either a free_text or review column")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    questions, secondary_topics = build_questions()
    rows = reviews.to_dict(orient="records")
    jobs = [(index, str(row[text_column]).strip()) for index, row in enumerate(rows) if str(row[text_column]).strip()]
    results: list[dict[str, Any] | None] = [None] * len(jobs)
    semaphore = asyncio.Semaphore(concurrency)
    retry = RetryPolicy(max_retries=max_retries, timeout=timeout)

    async with AsyncTypeSafeClient(api_key=api_key, model=model, timeout=timeout, retry=retry) as client:
        async def run_one(job_index: int, text: str) -> tuple[int, dict[str, Any]]:
            async with semaphore:
                try:
                    result = await analyze_one(
                        client,
                        text,
                        questions=questions,
                        secondary_topics=secondary_topics,
                        timeout=timeout,
                        secondary_threshold=secondary_threshold,
                    )
                except Exception as error:
                    # One transient API failure must not discard the whole batch.
                    result = {"jev_error": f"{type(error).__name__}: {error}"}
                return job_index, result

        tasks = [asyncio.create_task(run_one(index, text)) for index, text in enumerate(jobs)]
        completed = asyncio.as_completed(tasks)
        with tqdm(total=len(tasks), desc="Analyzing Jev reviews", unit="review") as progress:
            for completed_count, task in enumerate(completed, start=1):
                index, result = await task
                results[index] = result
                row_index, _ = jobs[index]
                rows[row_index].update(result)
                if completed_count % CHECKPOINT_EVERY == 0:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(rows).to_csv(output_path, index=False)
                    progress.set_postfix_str(f"checkpoint saved: {completed_count:,}")
                progress.update(1)

    for (row_index, _), result in zip(jobs, results):
        if result is not None:
            rows[row_index].update(result)
    enriched = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False)
    failed = int(enriched.get("jev_error", pd.Series(dtype=str)).fillna("").astype(bool).sum())
    print(f"Wrote {len(enriched):,} rows to {output_path}")
    print("Added column: jev_raw_output_json")
    if failed:
        print(f"Completed with {failed:,} failed reviews; see the jev_error column and rerun those rows.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--secondary-threshold", type=float, default=0.6)
    args = parser.parse_args()

    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        parser.error("TYPESAFE_API_KEY must be set")
    asyncio.run(
        analyze_file(
            input_path=args.input,
            output_path=args.output,
            api_key=api_key,
            model=args.model,
            timeout=args.timeout,
            max_retries=args.max_retries,
            concurrency=args.concurrency,
            secondary_threshold=args.secondary_threshold,
        )
    )


if __name__ == "__main__":
    main()

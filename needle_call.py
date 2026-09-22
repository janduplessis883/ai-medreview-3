from typing import Literal
from pathlib import Path

import pandas as pd
import needle


reviews = pd.read_csv("data/new_data_jev_analyzed.csv")

TOPIC_ALIASES = {
    "appointment access": "Appointment Availability and Lead Times",
    "appointment availability": "Appointment Availability and Lead Times",
    "phone access": "Telephone Access",
    "reception": "Reception Service",
}


def normalise(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    return " ".join(
        str(value)
        .lower()
        .replace("-", " ")
        .split()
    )



@needle.tool
def find_reviews(
    pcn: str | None = None,
    surgery: str | None = None,
    sentiment: Literal["Positive", "Negative", "Neutral or mixed"] | None = None,
    topic: str | None = None,
    minimum_urgency: float = 0.0,
) -> dict:
    """Search patient reviews.

    Args:
        pcn: The PCN name, for example Brompton-Health-PCN.
        surgery: The surgery name only. Never put sentiment or topic here.
        sentiment: The sentiment filter.
        topic: The primary service topic.
        minimum_urgency: Minimum urgency score from 0 to 3.
    """

    result = reviews.copy()

    if pcn:
        pcn_query = normalise(pcn)
        result = result[
            result["pcn"].map(normalise).str.contains(pcn_query, na=False)
        ]



    result = result[
        result["jev_primary_topic"]
        .map(normalise)
        .str.contains(topic_query, na=False)
    ]

    if surgery:
        result = result[result["surgery"].str.contains(surgery, case=False, na=False)]

    if sentiment:
        result = result[result["jev_sentiment"].eq(sentiment)]

    if topic:
        topic_query = normalise(topic)
        topic_query = TOPIC_ALIASES.get(topic_query, topic_query)


    result = result[result["jev_urgency"] >= minimum_urgency]

    return {
        "count": len(result),
        "reviews": result[
            ["date", "pcn", "surgery", "jev_sentiment", "jev_primary_topic", "review"]
        ].head(25).to_dict(orient="records"),
    }


agent = needle.Needle(
    weights="needle/needle3.cact",
    tools=[find_reviews],
)

response = agent.run(
    "Show me urgent negative reviews about appointment access at Brompton Health PCN."
)

print(response)

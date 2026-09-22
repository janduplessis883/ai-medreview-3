import json
from django.shortcuts import render
from .services import chart_colors, filter_reviews, load_reviews, jsonable_review


def dashboard(request):
    df = load_reviews()
    filtered = filter_reviews(df, request.GET)
    sentiment = filtered["jev_sentiment"].value_counts().to_dict()
    topics = filtered["jev_primary_topic"].value_counts().head(10).sort_values().to_dict()
    monthly = filtered.assign(month=filtered["review_date"].dt.strftime("%Y-%m")).groupby("month").size().to_dict()
    pcn_rows = []
    for pcn, group in filtered.groupby("pcn", dropna=False):
        pcn_rows.append({"pcn": pcn, "reviews": len(group), "positive": group["jev_sentiment"].eq("Positive").mean() * 100, "negative": group["jev_sentiment"].eq("Negative").mean() * 100, "actionable": (group["jev_actionability"] >= 2).mean() * 100, "review_queue": int(group["needs_review"].sum())})
    surgery_rows = []
    for (pcn, surgery), group in filtered.groupby(["pcn", "surgery"], dropna=False):
        surgery_rows.append({"pcn": pcn, "surgery": surgery, "reviews": len(group), "positive": group["jev_sentiment"].eq("Positive").mean() * 100, "negative": group["jev_sentiment"].eq("Negative").mean() * 100, "avg_actionability": group["jev_actionability"].mean(), "review_queue": int(group["needs_review"].sum())})
    queue = filtered[filtered["needs_review"]].sort_values(["jev_urgency", "jev_actionability"], ascending=False)
    selected_id = request.GET.get("review_id")
    selected = next((jsonable_review(row) for _, row in filtered.iterrows() if str(row["review_id"]) == selected_id), None)
    context = {
        "df": df, "filtered": filtered, "total_count": len(df), "selected": selected, "queue": queue.head(100).to_dict("records"),
        "pcn_rows": pcn_rows, "surgery_rows": surgery_rows, "sentiment_json": json.dumps(sentiment),
        "topics_json": json.dumps(topics), "monthly_json": json.dumps(monthly), "colors_json": json.dumps(chart_colors()),
        "pcns": sorted(df["pcn"].unique()), "surgeries": sorted(df["surgery"].unique()),
        "sentiments": sorted(df["jev_sentiment"].unique()), "topics": sorted(df["jev_primary_topic"].unique()),
        "data_source": str(__import__("reviews.services", fromlist=["data_path"]).data_path().name),
    }
    return render(request, "reviews/dashboard.html", context)

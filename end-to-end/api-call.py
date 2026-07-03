import math
import requests
from typing import Any, Dict, Optional


def get_best_source_link(api_url: str, query_params: Dict[str, Any]) -> Optional[str]:
    """Queries api.py, aggregates multiple chunk matches, and returns

    the Sefaria URL of the most likely source text.
    """
    # Over-fetch with unique=False to capture structural density across chunks
    payload = {
        "query": query_params.get("query"),
        "top_k": query_params.get("top_k", 50),
        "book_filter": query_params.get("book_filter"),
        "category_filter": query_params.get("category_filter"),
        "unique": False,
        "weighted": query_params.get("weighted", True),
        "addresses": query_params.get("addresses", [-1, -1, -1, -1]),
    }

    try:
        response = requests.post(f"{api_url.rstrip('/')}/search", json=payload)
        response.raise_for_status()
        results = response.json().get("results", [])
    except requests.RequestException:
        return None

    if not results:
        return None

    # Aggregate scores and track the URL for each sefaria_ref
    ref_groups = {}
    for item in results:
        ref = item["sefaria_ref"]
        if ref not in ref_groups:
            ref_groups[ref] = {
                "scores": [],
                "count": 0,
                "url": item["sefaria_url"],  # Grab the URL format from the API
            }
        ref_groups[ref]["scores"].append(item["score"])
        ref_groups[ref]["count"] += 1

    # Score and find the absolute winner
    best_url = None
    highest_confidence = -1.0

    for group in ref_groups.values():
        mean_score = sum(group["scores"]) / group["count"]
        # Add a small log bonus for multiple matching blocks in the same source
        confidence_score = mean_score + (0.05 * math.log1p(group["count"]))

        if confidence_score > highest_confidence:
            highest_confidence = confidence_score
            best_url = group["url"]

    return best_url


def main():
    api = "http://localhost:8000"
    params = {
        "query": "an ox that gores a donkey is considered to do so by habit",
        "top_k": 50,
        "book_filter": "",
        "category_filter": "",
        "addresses": [-1, -1, -1, -1],
        "unique": False,
        "weighted": True,
    }
    print(get_best_source_link(api_url=api, query_params=params))


if __name__ == "__main__":
    main()

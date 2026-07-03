from typing import Any


def build_search_queries(sentences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert LLM output into search query payloads.

    Input:
        [
            {
                "sentence_id": ...,
                "original_sentence": "...",
                "sources_found": [
                    {
                        "category": "...",
                        "book": "...",
                        "addr1": ...,
                        "addr2": ...,
                        "text": "..."
                    },
                    ...
                ]
            },
            ...
        ]

    Output:
        [
            {
                "sentence_id": ...,
                "queries": [
                    {
                        "query": "...",
                        "top_k": 10,
                        "book_filter": "...",
                        "category_filter": "...",
                        "unique": False,
                        "weighted": True,
                        "addresses": [addr1, addr2, -1, -1]
                    },
                    ...
                ]
            },
            ...
        ]
    """

    output = []

    for sentence in sentences:
        queries = []

        for source in sentence.get("sources_found", []):
            text = source.get("text")

            # Skip sources that don't require an API search.
            if text is None:
                continue

            queries.append(
                {
                    "query": text,
                    "top_k": 10,
                    "book_filter": source.get("book"),
                    "category_filter": source.get("category"),
                    "unique": False,
                    "weighted": True,
                    "addresses": [
                        source.get("addr1") if source.get("addr1") is not None else -1,
                        source.get("addr2") if source.get("addr2") is not None else -1,
                        -1,
                        -1,
                    ],
                }
            )

        if queries:
            output.append(
                {
                    "sentence_id": sentence["sentence_id"],
                    "queries": queries,
                }
            )

    return output
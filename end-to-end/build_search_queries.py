from typing import Any


def build_search_query(source: dict[str, Any]) -> dict[str, Any]:
    """Convert LLM output into search query payloads safely handling None values."""
    # 1. Safely handle book and category strings
    book = source.get("book") or ""
    category = source.get("category")

    # 2. Extract raw addresses
    raw_addr1 = source.get("addr1")
    raw_addr2 = source.get("addr2")

    # 3. Process addr1 safely
    if raw_addr1 is not None:
        if "talmud" in book.lower():
            # Ensure raw_addr1 is a string before passing to daf_to_talmud_index
            addr1 = daf_to_talmud_index(str(raw_addr1))
        else:
            addr1 = raw_addr1 + 1
    else:
        addr1 = -1

    # 4. Process addr2 safely
    if raw_addr2 is not None:
        addr2 = raw_addr2 + 1
    else:
        addr2 = -1

    # 5. Construct payload
    output = {
        "query": source.get("text"),
        "top_k": 10,
        "book_filter": source.get("book"),
        "category_filter": category,
        "unique": False,
        "weighted": True,
        "addresses": [addr1, addr2, -1, -1],
    }

    return output


def daf_to_talmud_index(amud: str) -> int:
    # Quick guard for empty or invalid strings
    if not amud or len(amud) < 2:
        return -1

    try:
        daf = int(amud[:-1])
        side = amud[-1].lower()
        return (daf - 2) * 2 + (2 if side == "b" else 1)
    except (ValueError, IndexError):
        return -1


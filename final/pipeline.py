"""
pipeline.py

Unified shiur-transcript source-resolution pipeline (v2).

Key design decisions, per review feedback:

  1. The transliteration engine converts Latin transliterations of Hebrew
     terms INTO Hebrew (not the other way around). So there is only ONE
     canonical text to run everything on downstream: `normalized_text`,
     the engine's output. There is no separate "Hebrew original" vs
     "transliterated" track anymore -- that was backwards in v1.

  2. FuzzyMatch is already functional and NOT something this file tries to
     improve. It is only fed the correctly-normalized text.

  3. The classifier flags BOTH actual quotes/paraphrases AND meta-mentions
     like "the Rambam says". Those are two different *roles* a flagged
     sentence can play:
       - LOCATOR: introduces/refreshes an AUTHORITY_OR_SOURCE / SOURCE_PATH
         / BOOK_LOCATION. May or may not carry quoted content itself.
       - CONTENT: the sentence that actually carries the quote/paraphrase
         to be searched against the corpus.
     A sentence can be both. Locator entities are held in a decaying
     context window so that content several sentences later can still
     resolve against them ("wanders for a bit" case).

  4. Book/author resolution goes through the DB directly (book_aliases,
     authors, author_book_map) rather than fuzzy-guessing against title
     strings alone. This also lets an author-only mention ("the Rambam")
     get narrowed to one specific book via the SOURCE_PATH text matched
     against that author's books.

  5. Resolved book/address is used to BOOST candidates from an
     unconstrained semantic search, never to hard-filter the initial
     query. A wrong book/alias resolution can therefore still surface the
     right passage; it just won't get the confidence bump.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

import requests

from HebrewTransliterationEngine import HebrewTransliterationEngine
from CitationClassifier import CitationClassifier
from FuzzyMatch import SefariaEngineV5
from ner_extractor import NERExtractor, convert_location_to_addresses

log = logging.getLogger("source_pipeline")
logging.basicConfig(level=logging.INFO)

DB_PATH = os.getenv("DB_PATH", "sefaria.db")
SEFARIA_API_URL = os.getenv("SEFARIA_API_URL", "http://localhost:8000")

# How many sentences a locator's context stays "live" for without being
# reinforced. Tune against real transcripts -- your NCSY Kollel sample
# wanders roughly 3-5 sentences between naming the Rambam and giving the
# actual halacha, so this starts a bit above that.
CONTEXT_WINDOW_SENTENCES = 6

# Confidence decays with distance from the sentence that supplied context.
# distance 0 (same sentence) -> no penalty.
CONTEXT_DISTANCE_PENALTY_PER_SENTENCE = 0.06

HEBREW_CHUNK_MIN_WORDS = 5

# Combined-score floor for the boosted merge. Kept lower than a pure
# unconstrained threshold would need to be, since corroborating signals
# (book match, address match) are folded into the same score rather than
# used as a gate.
SEMANTIC_ACCEPT_FLOOR = 0.40
# Even with boosts, don't accept something whose raw semantic score alone
# is noise -- boosts should tip a close call, not rescue a bad one.
RAW_SCORE_MIN = 0.22

BOOK_MATCH_BOOST = 0.18
ADDRESS_MATCH_BOOST = 0.15
ALIAS_EXACT_BOOST_BONUS = 0.05  # on top of BOOK_MATCH_BOOST for exact alias hits


# --------------------------------------------------------------------------
# Text normalization helpers
# --------------------------------------------------------------------------

_PUNCT_STRIP = re.compile(r"[\"'׳״`.,;:()\[\]]")


def normalize_title(s: str) -> str:
    if not s:
        return ""
    s = _PUNCT_STRIP.sub("", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def split_into_sentences(text: str) -> list[dict]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [
        {"index": i, "text": s.strip()} for i, s in enumerate(sentences) if s.strip()
    ]


def extract_hebrew_chunks(
    text: str, min_words: int = HEBREW_CHUNK_MIN_WORDS
) -> list[str]:
    hebrew_pattern = r"[\u0590-\u05FF]+(?:\s+[\u0590-\u05FF]+)*"
    raw_chunks = re.findall(hebrew_pattern, text)
    return [c for c in raw_chunks if len(c.split()) >= min_words]


def hebrew_density(text: str) -> float:
    letters = re.findall(r"[^\W\d_]", text, re.UNICODE)
    if not letters:
        return 0.0
    hebrew = [c for c in letters if "\u0590" <= c <= "\u05ff"]
    return len(hebrew) / len(letters)


# --------------------------------------------------------------------------
# Daf (Talmud-style) addressing
# --------------------------------------------------------------------------

DAF_PATTERN = re.compile(r"(?P<daf>\d+)\s*(?P<amud>[abᵃᵇ])?", re.IGNORECASE)


def daf_to_addr1(text: str) -> Optional[int]:
    """2a=1, 2b=2, 3a=3, 3b=4, ... (standard Vilna pagination starts at 2a).
    VERIFY against a known Gemara book's actual passages.addr1 values before
    relying on this -- can't confirm the offset without live DB access to
    real rows."""
    t = text.lower().replace("amud bet", "b").replace("amud aleph", "a")
    t = t.replace("amud a", "a").replace("amud b", "b")
    m = DAF_PATTERN.search(t)
    if not m or not m.group("daf"):
        return None
    daf = int(m.group("daf"))
    amud = "b" if (m.group("amud") or "a").lower() in ("b", "ᵇ") else "a"
    if daf < 2:
        return None
    return (daf - 2) * 2 + (1 if amud == "a" else 2)


def parse_location(
    location_text: Optional[str], source_path_text: Optional[str], book: Optional[dict]
) -> Optional[list]:
    """Turn NER's BOOK_LOCATION/SOURCE_PATH text into [addr1..addr4],
    using the resolved book's own address_types when we have one so a
    Talmud-style book gets the Daf formula and everything else gets plain
    positional numeric parsing. Returns None if there's nothing to parse."""
    text = location_text or source_path_text
    if not text:
        return None

    address_types = []
    if book and book.get("address_types"):
        try:
            address_types = json.loads(book["address_types"])
        except (TypeError, ValueError):
            address_types = []

    is_daf_scheme = bool(address_types) and address_types[0] in ("Talmud", "Daf")
    is_daf_scheme = is_daf_scheme or (
        source_path_text and "daf" in source_path_text.lower()
    )

    if is_daf_scheme:
        a1 = daf_to_addr1(text)
        return [a1 if a1 is not None else -1, -1, -1, -1]

    return convert_location_to_addresses(text)


def addresses_match(target: Optional[list], actual: Optional[list]) -> bool:
    if not target or not actual:
        return False
    for t, a in zip(target, actual):
        if t is None or t == -1:
            continue
        if a is None:
            a = -1
        if t != a:
            return False
    return any(t not in (None, -1) for t in target)


# --------------------------------------------------------------------------
# DB-backed catalog: aliases, authors, match templates
# --------------------------------------------------------------------------


class BookCatalog:
    """Direct read access to book_aliases / authors / author_book_map /
    books, so name resolution doesn't depend on guessing against the
    /books endpoint's LIKE search. Uses a read-only connection since this
    process should never need to write to the catalog."""

    def __init__(self, db_path: str = DB_PATH):
        self.conn = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, check_same_thread=False
        )
        self.conn.row_factory = sqlite3.Row

    def _book_row_to_dict(self, row: sqlite3.Row) -> dict:
        return {k: row[k] for k in row.keys()}

    def resolve_book_candidates(self, text: str, limit: int = 8) -> list[dict]:
        """Resolve a name against books.en_title/he_title AND book_aliases
        (both 'title' and 'match_template' alias types). Returns ranked
        candidates with a match_score and match_kind so the caller can
        tell an exact alias hit from a fuzzy LIKE fallback."""
        if not text or not text.strip():
            return []
        norm = normalize_title(text)
        if not norm:
            return []

        rows = self.conn.execute(
            """
            SELECT b.id, b.en_title, b.he_title, b.categories, b.address_types,
                   b.section_names, b.he_section_names,
                   NULL AS alias, 'canonical' AS alias_type
            FROM books b
            WHERE lower(b.en_title) LIKE ? OR lower(b.he_title) LIKE ?
            UNION ALL
            SELECT b.id, b.en_title, b.he_title, b.categories, b.address_types,
                   b.section_names, b.he_section_names,
                   ba.alias, ba.alias_type
            FROM book_aliases ba
            JOIN books b ON b.id = ba.book_id
            WHERE lower(ba.alias) LIKE ?
            """,
            (f"%{norm}%", f"%{norm}%", f"%{norm}%"),
        ).fetchall()

        candidates = []
        for r in rows:
            d = self._book_row_to_dict(r)
            compare_to = d["alias"] or d["en_title"] or d["he_title"] or ""
            score = difflib.SequenceMatcher(
                None, norm, normalize_title(compare_to)
            ).ratio()
            exact = normalize_title(compare_to) == norm
            if exact:
                score = 1.0
            d["match_score"] = score
            d["match_exact"] = exact
            candidates.append(d)

        # dedupe by book id, keep best score per book
        best_by_book: dict[str, dict] = {}
        for c in candidates:
            existing = best_by_book.get(c["id"])
            if existing is None or c["match_score"] > existing["match_score"]:
                best_by_book[c["id"]] = c

        ranked = sorted(
            best_by_book.values(), key=lambda c: c["match_score"], reverse=True
        )
        return ranked[:limit]

    def resolve_authors(self, text: str, limit: int = 5) -> list[dict]:
        if not text or not text.strip():
            return []
        norm = normalize_title(text)
        rows = self.conn.execute(
            """
            SELECT id, en_name, he_name FROM authors
            WHERE lower(en_name) LIKE ? OR lower(he_name) LIKE ?
            """,
            (f"%{norm}%", f"%{norm}%"),
        ).fetchall()
        candidates = []
        for r in rows:
            d = self._book_row_to_dict(r)
            name = d["en_name"] or d["he_name"] or ""
            score = difflib.SequenceMatcher(None, norm, normalize_title(name)).ratio()
            d["match_score"] = 1.0 if normalize_title(name) == norm else score
            candidates.append(d)
        return sorted(candidates, key=lambda c: c["match_score"], reverse=True)[:limit]

    def books_by_author(self, author_id: str) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT b.id, b.en_title, b.he_title, b.categories, b.address_types,
                   b.section_names, b.he_section_names
            FROM author_book_map m
            JOIN books b ON b.id = m.book_id
            WHERE m.author_id = ?
            """,
            (author_id,),
        ).fetchall()
        return [self._book_row_to_dict(r) for r in rows]

    def resolve(
        self, authority_text: Optional[str], source_path_text: Optional[str]
    ) -> dict:
        """Top-level resolution combining direct book/alias matches with
        author -> books narrowing. Returns:
          {"candidates": [book_dict, ...] (best first), "method": str}
        `candidates` is intentionally a list, not a single answer -- the
        caller uses these as *boost* targets for semantic search, not as
        a hard filter, so ambiguity here is fine as long as the true book
        is somewhere in the list.
        """
        # 1. Try the most specific text first (SOURCE_PATH usually names
        #    the actual work, e.g. "Hilchos Avel", more precisely than a
        #    bare author name).
        for candidate_text, label in (
            (source_path_text, "source_path"),
            (authority_text, "authority"),
        ):
            books = (
                self.resolve_book_candidates(candidate_text) if candidate_text else []
            )
            good = [b for b in books if b["match_score"] >= 0.6]
            if good:
                return {"candidates": good, "method": f"alias:{label}"}

        # 2. Fall back to author -> book(s), narrowed by source_path terms
        #    if the author wrote more than one work.
        if authority_text:
            authors = self.resolve_authors(authority_text)
            authors = [a for a in authors if a["match_score"] >= 0.6]
            if authors:
                all_books: list[dict] = []
                for a in authors[:2]:
                    all_books.extend(self.books_by_author(a["id"]))
                if len(all_books) <= 1:
                    for b in all_books:
                        b["match_score"] = authors[0]["match_score"]
                        b["match_exact"] = False
                    return {"candidates": all_books, "method": "author_single_work"}

                # multiple works by this author -- narrow using source_path
                # text against each candidate book's own aliases/sections
                narrowed = []
                if source_path_text:
                    sp_norm = normalize_title(source_path_text)
                    for b in all_books:
                        text_pool = " ".join(
                            filter(
                                None,
                                [
                                    b.get("section_names") or "",
                                    b.get("he_section_names") or "",
                                    b.get("en_title") or "",
                                    b.get("he_title") or "",
                                ],
                            )
                        )
                        score = difflib.SequenceMatcher(
                            None, sp_norm, normalize_title(text_pool)
                        ).ratio()
                        if score >= 0.25:
                            b = dict(b)
                            b["match_score"] = score
                            narrowed.append(b)
                if narrowed:
                    narrowed.sort(key=lambda b: b["match_score"], reverse=True)
                    return {
                        "candidates": narrowed,
                        "method": "author_narrowed_by_source_path",
                    }

                # couldn't narrow -- return all of them as low-confidence
                # candidates rather than picking one blindly
                for b in all_books:
                    b["match_score"] = 0.3
                return {"candidates": all_books, "method": "author_ambiguous"}

        return {"candidates": [], "method": "none"}


# --------------------------------------------------------------------------
# Semantic search client (api_v3.py FastAPI service)
# --------------------------------------------------------------------------


class SefariaClient:
    def __init__(self, base_url: str = SEFARIA_API_URL, timeout: float = 200.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _search_raw(
        self,
        query: str,
        top_k: int,
        book_filter: Optional[str],
        addresses: Optional[list],
    ) -> list[dict]:
        payload = {
            "query": query,
            "top_k": top_k,
            "book_filter": book_filter,
            "addresses": addresses or [-1, -1, -1, -1],
            "rerank": True,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/search", json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json().get("results", [])
        except requests.RequestException as e:
            log.warning(
                "Search failed (book_filter=%r) for query=%r: %s", book_filter, query, e
            )
            return []

    def resolve_content(
        self,
        query: str,
        candidate_books: list[dict],
        target_address: Optional[list],
        top_k: int = 8,
    ) -> list[dict]:
        """Boost-not-filter merge: always run an unconstrained search, then
        separately run scoped searches against the top candidate books, and
        merge everything by sefaria_ref. Book/address agreement raises the
        combined score; it never removes a candidate the unconstrained pass
        already surfaced. This is deliberate -- a wrong alias/author
        resolution should degrade confidence, not silently drop the right
        answer from consideration."""
        merged: dict[str, dict] = {}

        for r in self._search_raw(query, top_k=15, book_filter=None, addresses=None):
            ref = r["sefaria_ref"]
            r["combined_score"] = r["score"]
            r["book_boosted"] = False
            r["address_boosted"] = False
            merged[ref] = r

        for book in candidate_books[:3]:
            book_id = book.get("id")
            if not book_id:
                continue
            for r in self._search_raw(
                query, top_k=top_k, book_filter=book_id, addresses=target_address
            ):
                ref = r["sefaria_ref"]
                boost = BOOK_MATCH_BOOST + (
                    ALIAS_EXACT_BOOST_BONUS if book.get("match_exact") else 0.0
                )
                actual_addr = [
                    r.get("addr1"),
                    r.get("addr2"),
                    r.get("addr3"),
                    r.get("addr4"),
                ]
                addr_hit = addresses_match(target_address, actual_addr)
                if addr_hit:
                    boost += ADDRESS_MATCH_BOOST

                existing = merged.get(ref)
                if existing is None:
                    r["combined_score"] = r["score"] + boost
                    r["book_boosted"] = True
                    r["address_boosted"] = addr_hit
                    merged[ref] = r
                else:
                    existing["combined_score"] = max(
                        existing["combined_score"], existing["score"] + boost
                    )
                    existing["book_boosted"] = True
                    existing["address_boosted"] = (
                        existing["address_boosted"] or addr_hit
                    )

        results = list(merged.values())
        results.sort(key=lambda r: r["combined_score"], reverse=True)
        return results


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass
class ResolvedSource:
    status: str  # "fuzzy_match" | "resolved" | "unresolved"
    confidence: float
    sefaria_ref: Optional[str] = None
    en_title: Optional[str] = None
    he_title: Optional[str] = None
    book_id: Optional[str] = None
    text: Optional[str] = None
    sefaria_url: Optional[str] = None
    addr: Optional[list] = None
    resolution_method: Optional[str] = (
        None  # alias:source_path / author_narrowed_by_source_path / etc.
    )
    context_distance: int = 0  # sentences back the locator entity came from
    book_boosted: bool = False
    address_boosted: bool = False
    raw_authority: Optional[str] = None
    raw_location: Optional[str] = None
    raw_source_path: Optional[str] = None
    note: Optional[str] = None


@dataclass
class SentenceResult:
    index: int
    text: str
    is_citation: bool
    role: str = "none"  # "locator" | "content" | "both" | "none"
    sources: list = field(default_factory=list)


@dataclass
class _LocatorContext:
    authority_text: Optional[str]
    location_text: Optional[str]
    source_path_text: Optional[str]
    origin_index: int
    age: int = 0


# --------------------------------------------------------------------------
# Resolver
# --------------------------------------------------------------------------


class SourceResolver:
    def __init__(self, db_path: str = DB_PATH, sefaria_base_url: str = SEFARIA_API_URL):
        self.transliterator = HebrewTransliterationEngine()
        self.classifier = CitationClassifier()
        self.ner = NERExtractor()
        self.catalog = BookCatalog(db_path)
        self.client = SefariaClient(sefaria_base_url)

    def run(self, transcript: str) -> list[SentenceResult]:
        normalized_text = self.transliterator.process_transcript(transcript)
        sentences = split_into_sentences(normalized_text)

        citation_flags = self.classifier.classify_sentences(sentences)
        citation_indices = self._normalize_citation_flags(citation_flags, sentences)

        # NER on every sentence, not just flagged ones -- a locator can sit
        # in a sentence the classifier didn't flag as a citation at all.
        entities_by_index = {
            s["index"]: self.ner.process_sentence(s["text"]) for s in sentences
        }

        results: list[SentenceResult] = []
        context: Optional[_LocatorContext] = None

        for s in sentences:
            idx = s["index"]
            text = s["text"]
            ents = entities_by_index[idx]
            is_citation = idx in citation_indices

            authority = (
                ents["AUTHORITY_OR_SOURCE"][0] if ents["AUTHORITY_OR_SOURCE"] else None
            )
            source_path = ents["SOURCE_PATH"][0] if ents["SOURCE_PATH"] else None
            location = ents["BOOK_LOCATION"][0] if ents["BOOK_LOCATION"] else None
            has_locator_entities = bool(authority or source_path)

            hchunks = extract_hebrew_chunks(text)
            looks_like_content = bool(hchunks) or (
                is_citation and not has_locator_entities
            )

            role = "none"
            if has_locator_entities and looks_like_content:
                role = "both"
            elif has_locator_entities:
                role = "locator"
            elif is_citation or looks_like_content:
                role = "content"

            # -- update / start context on locator signal --
            if has_locator_entities:
                is_new_topic = (
                    context is None
                    or (
                        authority
                        and normalize_title(authority)
                        != normalize_title(context.authority_text or "")
                    )
                    or (
                        source_path
                        and normalize_title(source_path)
                        != normalize_title(context.source_path_text or "")
                    )
                )
                if is_new_topic:
                    context = _LocatorContext(
                        authority_text=authority,
                        location_text=location,
                        source_path_text=source_path,
                        origin_index=idx,
                        age=0,
                    )
                else:
                    context.age = 0
                    context.location_text = location or context.location_text
                    context.source_path_text = source_path or context.source_path_text
            elif (
                location
                and context is not None
                and context.age <= CONTEXT_WINDOW_SENTENCES
            ):
                # a bare location mention ("perek daled") reinforcing an
                # already-open context, without re-naming the authority
                context.location_text = location
                context.age = 0

            sr = SentenceResult(
                index=idx, text=text, is_citation=is_citation, role=role
            )

            if role in ("content", "both") and (is_citation or hchunks):
                sr.sources = self._resolve_content_sentence(
                    text=text,
                    local_authority=authority,
                    local_source_path=source_path,
                    local_location=location,
                    context=context if not has_locator_entities else None,
                )

            results.append(sr)

            if context is not None:
                context.age += 1
                if context.age > CONTEXT_WINDOW_SENTENCES:
                    context = None

        return results

    @staticmethod
    def _normalize_citation_flags(classified, all_sentences) -> set:
        if classified and isinstance(classified[0], dict):
            return {s["index"] for s in classified}
        return {s["index"] for s, flag in zip(all_sentences, classified) if flag}

    def _resolve_content_sentence(
        self,
        text: str,
        local_authority: Optional[str],
        local_source_path: Optional[str],
        local_location: Optional[str],
        context: Optional[_LocatorContext],
    ) -> list[ResolvedSource]:
        sources: list[ResolvedSource] = []

        fuzzy = SefariaEngineV5()
        # --- Tier 1: fuzzy exact match (untouched, already functional) ---
        for chunk in extract_hebrew_chunks(text):
            ref = fuzzy.execute_search(chunk)
            if ref:
                sources.append(
                    ResolvedSource(
                        status="fuzzy_match", confidence=0.95, **self._ref_fields(ref)
                    )
                )

        # decide which locator info to use: sentence-local beats context
        authority = local_authority or (context.authority_text if context else None)
        source_path = local_source_path or (
            context.source_path_text if context else None
        )
        location = local_location or (context.location_text if context else None)
        distance = (
            0
            if (local_authority or local_source_path)
            else (context.age if context else 0)
        )

        if not authority and not source_path:
            if not sources:
                sources.append(
                    ResolvedSource(
                        status="unresolved",
                        confidence=0.0,
                        note="Looked like a citation but no AUTHORITY_OR_SOURCE/SOURCE_PATH "
                        "found locally or in the recent context window.",
                    )
                )
            return sources

        resolution = self.catalog.resolve(authority, source_path)
        candidates = resolution["candidates"]
        target_address = None
        if candidates:
            target_address = parse_location(location, source_path, candidates[0])
        elif location or source_path:
            target_address = parse_location(location, source_path, None)

        merged = self.client.resolve_content(text, candidates, target_address)

        distance_penalty = distance * CONTEXT_DISTANCE_PENALTY_PER_SENTENCE
        accepted = [
            r
            for r in merged
            if r["score"] >= RAW_SCORE_MIN
            and (r["combined_score"] - distance_penalty) >= SEMANTIC_ACCEPT_FLOOR
        ]

        if accepted:
            best = accepted[0]
            sources.append(
                ResolvedSource(
                    status="resolved",
                    confidence=round(
                        min(0.97, best["combined_score"] - distance_penalty), 3
                    ),
                    resolution_method=resolution["method"],
                    context_distance=distance,
                    book_boosted=best.get("book_boosted", False),
                    address_boosted=best.get("address_boosted", False),
                    raw_authority=authority,
                    raw_location=location,
                    raw_source_path=source_path,
                    **self._ref_fields(best),
                )
            )
        else:
            note = "No passage cleared the confidence floor."
            if not candidates:
                note = (
                    "No book/author alias matched this name in the catalog -- "
                    "likely an alternate title, or a book not in the corpus. "
                    "Flag for catalog review."
                )
            sources.append(
                ResolvedSource(
                    status="unresolved",
                    confidence=0.0,
                    resolution_method=resolution["method"],
                    context_distance=distance,
                    raw_authority=authority,
                    raw_location=location,
                    raw_source_path=source_path,
                    note=note,
                )
            )

        return sources

    @staticmethod
    def _ref_fields(result: dict) -> dict:
        return dict(
            sefaria_ref=result.get("sefaria_ref"),
            en_title=result.get("en_title"),
            he_title=result.get("he_title"),
            book_id=result.get("book_id"),
            text=result.get("text"),
            sefaria_url=result.get("sefaria_url"),
            addr=[
                result.get("addr1"),
                result.get("addr2"),
                result.get("addr3"),
                result.get("addr4"),
            ]
            if "addr1" in result
            else None,
        )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def find_links(
    transcript: str, db_path: str = DB_PATH, sefaria_base_url: str = SEFARIA_API_URL
) -> list[SentenceResult]:
    return SourceResolver(db_path, sefaria_base_url).run(transcript)


def main():
    transcript = """
Good afternoon everybody. So a little bit of a lighter topic today and that is the topic of stripes on a <i>tallis</i>. You may notice that many people have black stripes on their <i>tallis</i>, some people have blue stripes on their <i>tallis</i>, some people have white stripes on their <i>tallis</i> and some people have Joseph's Technicolor Dreamcoat with all the different colors of stripes on the <i>tallis</i>. So the question is: where does this come from? Is this a <i>minhag Yisrael</i>? Is it just something that we do? So the source is actually three possibilities that are mentioned by the <i>sefarim</i>. There's a <i>Pri Megadim</i> in <i>Siman Tes</i> in <i>Hilchos Tzitzis</i> where he suggests that it's a <i>zecher l'techeiles</i>, that for many, many years the Jewish people didn't have <i>techeiles</i>. In fact, one of the more common questions I get when people get <i>techeiles</i> is, now that I have <i>techeiles</i>, should I not have stripes on my tallis? You're still allowed to have stripes on the <i>tallis</i>. But the idea was that it was to remember the <i>techeiles</i>. And that's why you'll often find even on a <i>tallis katan</i>, only a wool <i>tallis katan</i> will have black stripes, but a cotton <i>tallis katan</i> doesn't have black stripes. Now why would that be the case? Meaning, did they not figure out how to make black cotton? I mean, you have black t-shirts that are cotton. They figured it out, right? So why can't they make a black stripe on a cotton <i>tallis katan</i>? Because <i>techeiles</i> only works on wool. <i>Techeiles</i> is by definition wool. So if you're going to have black stripes, it's got to be on a wool <i>beged</i> if it's going to be a <i>zecher l'techeiles</i>. So that's possibility number one. Possibility number two is based on <i>Kabbalah</i>, is based on the <i>Zohar</i>, where the <i>Zohar</i> understands that black represents the <i>midas hagevura</i> and white represents the <i>midas hachessed</i> or <i>rachamim</i>. So there's this fusion of the <i>gevura</i> and the <i>rachamim</i> should be represented on the <i>tallis</i>. So therefore it's not just white, which is the primary color, but it's also the black or blue, which is the secondary color. And the third reason for the very concept of having stripes on the <i>tallis</i> is based on the <i>Mishna Masechet Berachot</i>. The <i>Mishna</i> talks about the different <i>zmanim</i> in the morning and when the earliest <i>zman</i> that you could say <i>Kriyat Shema</i> is. And it says to be מבחין בין תכלת ללבן. So how do you know if you're able to be מבחין בין תכלת ללבן? If you have it right in front of you and you're able to see that you have a white <i>beged</i> and a <i>techeiles</i> stripe on it, so then you could see if you're able to be מבחין בין תכלת ללבן. So maybe that's why it was included on the <i>tallis</i>, not only as a <i>zecher l'techeiles</i>, but to function for one of the it can't function as the <i>mitzva d'oraisa</i> of <i>techeiles</i>, but it could function as one of the things that <i>techeiles</i> helped us out in in terms of being able to tell when the <i>zman</i> is for <i>Kriyat Shema</i>. So now that we know three possible sources, again, either a <i>zecher l'techeiles</i> or <i>al pi Kabbalah</i> or to be מבחין בין תכלת ללבן, what is the basis for each of the different colors that you find? So the most common one you find nowadays is black. This could be based on the <i>Rambam</i> in <i>Hilchos Tzitzis</i> well, not really the <i>Rambam</i>. The <i>Rambam</i> held that in הלכות ציצית פרק ב that the <i>dam hachilazon</i> that they would use was a black <i>dam</i>, although the <i>Rambam</i> seems to understand that once it went through the process it actually produced a blue dye and not a black dye. The <i>Gemara Menachot</i> tells us that the <i>techeiles</i> is <i>domeh l'yam</i> and the <i>yam</i> is <i>domeh l'rakia</i>, so the <i>techeiles</i> in some way is supposed to be the color of the <i>yam</i> or the color of the <i>rakia</i>, but it doesn't say explicitly in the <i>Gemara</i> at what time of day. The <i>rakia</i> has a very different color during broad daylight than it does at nighttime. So <i>Rashi</i> seems to understand that it's the <i>rakia</i> toward evening time, which would mean that it would be a much darker blue. But the <i>Rambam</i> learns that it's the middle of the day. So that might be some of the idea of the black, that if you hold that the <i>techeiles</i> is supposed to be black, then you'd understand that the stripes are supposed to be black, or even if the <i>techeiles</i> are supposed to be a blackish color, a very, very dark blue that almost appears to be black. The other possibility is that the Torah itself is referred to as אש שחורה על גבי אש לבנה. So there's the sense of the <i>levanah</i>, of the white, which is purity and innocence, and the black lines or the letters of the Torah are the <i>eish sh'chora</i> on top of the <i>eish levanah</i>, and this is representative of that. Okay, so maybe in somehow it's representative of Torah because after all, the <i>tzitzis</i> themselves, <i>Rashi</i> in <i>Chumash</i> tells us, are supposed to represent all <i>taryag mitzvos</i>. The <i>Gemara</i> says that it's supposed to remind us of all <i>taryag mitzvos</i>, so maybe the <i>beged</i> upon which you're going to put the <i>tzitzis</i> should somehow represent the entirety of Torah. Now the idea of blue, I did not know this, but in doing a little bit of research, it seems that blue used to be the more common <i>minhag</i>, that it grew out of favor in more contemporary times, but it used to be the more common <i>minhag</i>. In fact, there was a prominent Zionist businessman, David Wolffsohn, who came to the he had to figure it was the design proposed by Theodor Herzl for the Israeli flag wasn't gaining any support and he wrote David Wolffsohn wrote that at the At the behest of our leader Herzl, I came to Basel to make preparations for the Zionist Congress. Among many other problems that occupied me then was one that contains something of the essence of the Jewish problem: What flag would we hang in the Congress hall? Then an idea struck me. We have a flag, and it's blue and white, the <i>talis</i> with which we wrap ourselves when we pray, that is our symbol. So all I have to do is take the <i>talis</i> out of the bag, slap a <i>Magen David</i> on it, and that's our flag. And that was the concept, that was the concept of the Israeli flag. So the assumption was that it was blue, that the <i>talis</i> had blue stripes on it. Where does the idea of white stripes come from? So certainly you see on <i>Yom Kippur</i> people wear white because people always wear white on <i>Yom Kippur</i>, it's a symbol of purity, you know, that <i>im chateichem</i> is <i>kashaleg yalbinu</i>, that even though we may have sins, we hope that it becomes whitened and purified. But even not on <i>Yom Kippur</i>, there seems to be a fairly common Sephardic practice to have white stripes on the <i>talis</i>. So where does that come from? So the <i>Birkei Yosef</i>, the <i>Chida</i> writes in <i>Birkei Yosef</i> in <i>Shiurei Bracha</i> in the name of the <i>Marsha</i> that <i>yesh omrim shetzarich</i> that the <i>talis</i> should be white, and it could be that it's partially based on what the <i>Shulchan Aruch</i> writes in <i>Siman Tes</i>, <i>Seif Heh</i>, where the <i>Shulchan Aruch</i> writes יש אומרים שצריך לעשות הציצית מצבע הטלית והמדקדקים נוהגים כן and the <i>Rama</i> writes ואשכנזים אינם נוהגים לעשות ציצית רק לבנים אף בבגדים צבועים אין לשנות. Now you put two things together: number one, that the <i>tzitzis</i> are supposed to match the <i>beged</i>, and number two, the <i>Rama</i> writes that our <i>minhag</i> is we only use white <i>tzitzis</i>. So if the <i>tzitzis</i> are supposed to match the <i>beged</i> and you only use white <i>tzitzis</i>, then the <i>beged</i> should be white. So the practice is to have white. It's also <i>al pi kabbalah</i>, I think, that a lot of Sephardim have <i>davka</i> fully white. The <i>Ben Ish Chai</i> writes, though, in <i>Parshas Noach</i>, that it doesn't bother him that there are stripes of a different color. He says יזהר לעשות הטליתות וחוטן של הציציות הכל לבן בין בטלית גדול בין בטלית קטן רמז לדבר ולבן שיניים מחלב וטליתות שיש בהן צבע מצדדין לנוי לא אכפת. He says if there's a little bit of color on the side to make it more beautiful, that doesn't bother me כי הצבע הוא מעט מן המעט בגוף הטלית שרובו לבן. It's mostly white, so that fulfills the <i>din</i>. So even though the more common Sephardic practice, I think, is to have white stripes, you will find pictures if you do a Google image search of famous Sephardi rabbis that I'm familiar with, so Rav בן ציון אבא שאול, there are pictures of him with a <i>talis</i> with black stripes on it. Rav <i>Kaduri</i>, there are pictures of him with a <i>talis</i> with black stripes on it. Rav <i>Ovadia</i>, mostly white, but you find from the early years some pictures with him with a <i>talis</i> with black stripes on it also. So it's not the end of the world. But this is a little bit about perhaps where some of this comes from.
    """
    results = find_links(
        transcript=transcript,
        db_path="/home/zelfinit/CS/sofer/sefaria/sefaria.db",
        sefaria_base_url="http://127.0.0.1:8000",
    )
    out = [
        {
            "index": r.index,
            "text": r.text,
            "is_citation": r.is_citation,
            "role": r.role,
            "sources": [s.__dict__ for s in r.sources],
        }
        for r in results
    ]
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

import sqlite3
import re

DB_FILE = "sefaria.db"

class SefariaEngineV5:
    def __init__(self, db_path=DB_FILE):
        self.conn = sqlite3.connect(db_path)
        # OPTIMIZED: Pointing to _v5 tables and fetching snippet text natively from FTS5
        # snippet(chunks_fts_v5, 0, '【', '】', '...', 15) AS text_match
        self.search_statement = """
            SELECT 
                f.rank,
                p.book_id,
                p.addr1,
                p.addr2,
                p.addr3,
                p.addr4,
                c.text,
                c.id,
                b.categories  
            FROM chunks_fts_v5 f
            JOIN sentences_lookup_v5 sl ON sl.id = f.rowid
            JOIN chunks c ON c.id = sl.passage_id
            JOIN passages p ON p.id = c.passage_id
            JOIN books b ON b.id = p.book_id
            WHERE chunks_fts_v5 MATCH ?
            ORDER BY f.rank ASC
            LIMIT ?;
        """
        #self.pagerank_scores = pagerank_dict if pagerank_dict else {}

    def execute_search(self, user_phrase, proximity_window=10, num_of_sources_to_tank=1000, num_of_results=10):
        """
        Executes a single fuzzy pass. Natively guarantees unique, deduplicated 
        text chunks and enforces strict limits instantly.
        """
        clean_phrase = re.sub(r'[*":^|-]', ' ', user_phrase).strip()
        words = clean_phrase.split()
        if not words:
            return []
            
        # Target unique root terms (words longer than 2 characters)
        stable_terms = [w for w in words if len(w) > 2]
        if not stable_terms:
            stable_terms = words  # Fallback for ultra-short queries
            
        # Combine using OR. This captures exact phrases and handles typos simultaneously.
        query_string = " OR ".join(f'"{w}"' for w in stable_terms)
        
        cursor = self.conn.cursor()
        # SQLite natively enforces uniqueness and the exact LIMIT right here
        cursor.execute(self.search_statement, (query_string, num_of_sources_to_tank))
        records = cursor.fetchall()
        cursor.close()

        formatted_results = []
        unique_matches = {}
        for bm25_rank, book_name, a1, a2, a3, a4, text_match, id, categories in records:
            location = (book_name, a1, a2, a3, a4)
            if location not in unique_matches:
                # FTS5 rank outputs negative numbers. Make it positive for easier scaling math.
                raw_score = abs(bm25_rank) if (bm25_rank and bm25_rank < 0) else 0.0
                    
                # Cache everything temporarily in memory for the re-scoring pass
                unique_matches[location] = {
                    "base_score": raw_score,
                    "book_id": book_name,
                    "categories": categories,
                    "coords": [a1, a2, a3, a4],
                    "text_match": text_match
                }

        raw_scores = [data["base_score"] for data in unique_matches.values()]
        max_score = max(raw_scores) if raw_scores else 0.0
        min_score = min(raw_scores) if raw_scores else 0.0
        score_range = max_score - min_score
        

        for location, data in unique_matches.items():
            if score_range == 0:
                normalized_score = 1.0 if max_score > 0 else 0.0
            else:
                normalized_score = (data["base_score"] - min_score) / score_range

            weighted_score = self.weightify(normalized_score, data["categories"], data["book_id"])

            proximity_factor = self._proximity_order_score(data["text_match"], stable_terms, proximity_window)
            weighted_score *= proximity_factor

            formatted_results.append([
                weighted_score,
                data["book_id"],
                data["text_match"],
                data["coords"],
                data["categories"]
            ])
        formatted_results.sort(key=lambda x: x[0], reverse=True)
        return formatted_results[:num_of_results]

    def close(self):
        self.conn.close()

    def _proximity_order_score(self, text, terms, proximity_window=60):
        """
        Scores how well the query terms that ARE present in `text` line up
        with the order they were typed, and how tightly clustered they are.
        Terms not found verbatim are skipped rather than disqualifying the
        row (they might be spelling variants / not indexed the same way).
 
        Returns a multiplier roughly in [0.5, 1.3]:
          - more matched terms -> higher
          - matched terms in the same relative order -> higher
          - matched terms close together (small span) -> higher
          - a row where NONE of the terms literally appear returns 0.0,
            since that means the match came purely from a totally
            unrelated OR branch.
        """
        if not text or not terms:
            return 0.5
 
        positions = []
        search_from = 0
        for term in terms:
            idx = text.find(term, search_from)
            if idx == -1:
                idx = text.find(term)  # allow out-of-order matches to still register
            if idx == -1:
                continue  # term not found verbatim - skip, don't disqualify
            positions.append(idx)
            search_from = idx + len(term)
 
        matched_count = len(positions)
        total_terms = len(terms)
 
        if matched_count == 0:
            # None of the query terms literally occur in this chunk's text -
            # it only matched via some other row's term through OR upstream
            # in a differently-deduped path. Treat as no real signal.
            return 0.0
 
        coverage_ratio = matched_count / total_terms
 
        if matched_count >= 2:
            in_order_pairs = sum(
                1 for i in range(matched_count - 1) if positions[i] <= positions[i + 1]
            )
            order_ratio = in_order_pairs / (matched_count - 1)
 
            span = max(positions) - min(positions)
            proximity_ratio = max(0.0, 1 - (span / max(proximity_window, 1)))
        else:
            order_ratio = 1.0
            proximity_ratio = 1.0
 
        # Coverage matters most (need the words there at all), then order,
        # then how tightly clustered they are.
        combined = (0.55 * coverage_ratio) + (0.25 * order_ratio) + (0.20 * proximity_ratio)
 
        # Rescale 0..1 -> 0.5..1.3 so weak-but-real matches get dinged,
        # not zeroed, while strong in-order/close matches get boosted.
        return 0.5 + (combined * 0.8)


    def weightify(self, score: float, categories: str, book_id: str) -> float:
        """Your exact categorical priority matching and PageRank multiplier formula."""
        if not categories:
            categories = ""
        categories_set = set(categories.split('|'))

        # 1. Immediate exit for 0-weight categories
        if categories_set.intersection({"Second Temple", "Reference"}):
            return 0.0

        base_weighted_score = score
        has_matched = True

        if "Tanakh" in categories_set:
            if "Rishonim on Tanakh" in categories_set: base_weighted_score = score * 1.05
            elif "Acharonim on Tanakh" in categories_set: base_weighted_score = score * 1.03
            elif "Modern Commentary on Tanakh" in categories_set: return 0.0
            elif "Targum" in categories_set: base_weighted_score = score * 1.04
            else: base_weighted_score = score * 1.1

        elif "Talmud" in categories_set:
            if "Bavli" in categories_set:
                if "Rishonim on Talmud" in categories_set: base_weighted_score = score * 1.03
                elif "Acharonim on Talmud" in categories_set: base_weighted_score = score * 1.01
                elif "Commentary on Minor Tractates" in categories_set: base_weighted_score = score * 0.95
                elif "Guides" in categories_set: return 0.0
                elif "Minor Tractates" in categories_set: base_weighted_score = score
                elif "Modern Commentary on Talmud" in categories_set: return 0.0
                else: base_weighted_score = score * 1.08
            else:
                base_weighted_score = score * 0.95 if "Commentary" in categories_set else score * 1.02

        elif "Tosefta" in categories_set or "Midrash" in categories_set:
            base_weighted_score = score * 0.95 if "Commentary" in categories else score

        elif "Halakha" in categories_set:
            if "Modern" in categories_set: base_weighted_score = score * 0.5
            elif "Commentary" in categories: base_weighted_score = score
            else: base_weighted_score = score * 1.04

        elif categories_set.intersection({"Responsa", "Musar", "Jewish Thought"}):
            if "Modern" in categories:
                return 0.0
            elif "Jewish Thought" in categories_set:
                base_weighted_score = score * 0.93
            else:
                base_weighted_score = score

        elif "Mishnah" in categories_set:
            if "Rishonim on Mishnah" in categories_set or "Acharonim on Mishnah" in categories_set: base_weighted_score = score * 1.01
            elif "Modern Commentary on Mishnah" in categories_set: return 0.0
            else: base_weighted_score = score * 1.06

        else:
            flat_weights = {"Liturgy": 0.98, "Kabbalah": 0.80, "Chasidut": 0.93}
            has_matched = False
            for cat in categories_set:
                if cat in flat_weights:
                    base_weighted_score = score * flat_weights[cat]
                    has_matched = True
                    break

            if not has_matched:
                base_weighted_score = score

        # 2. Dynamic PageRank fine-tuning overlay
        #if self.pagerank_scores:
        #    clean_book_id = book_id.strip()
        #    pr_score = self.pagerank_scores.get(clean_book_id, 0.0)
        #    pr_boost = 1.0 + (4.5 * pr_score)
        #    return base_weighted_score * pr_boost

        return base_weighted_score
    





if __name__ == "__main__":
    engine = SefariaEngineV5()
    
    test_phrase = "רבבה כצמח השדה נתתיך ותרבי ותגדלי ותבואי בעדי עדיים שדיים נכונו ושערך צמח ואת ערום ועריה"
    
    print("Running optimized long-query native trigram lookup against full chunks (v5)...")
    results = engine.execute_search(test_phrase, num_of_results=10)
    
    print("\nResults:")
    for row in results:
        print(f"Score: {row[0]} | Book ID: {row[1]} | {row[4]}")
        print(f"Match Context: {row[2]}")
        print(f"Coordinates: {row[3]}")
        #print("-" * 50)
        
    engine.close()

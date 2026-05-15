from rank_bm25 import BM25Okapi


DOMAIN_BOOSTS = [
    (
        ["leadership", "executive", "director", "cxo", "vp"],
        ["leadership", "executive", "director"],
        2.0
    ),
    (
        ["manager", "management"],
        ["manager", "management", "leadership"],
        1.5
    ),
    (
        ["java"],
        ["java"],
        2.0
    ),
    (
        ["python"],
        ["python"],
        2.0
    ),
    (
        ["sales"],
        ["sales"],
        2.0
    ),
    (
        ["communication", "stakeholder", "interpersonal"],
        ["communication", "interpersonal", "verbal"],
        1.5
    ),
    (
        ["personality", "behavior", "behaviour"],
        ["personality", "behavior", "behaviour"],
        1.5
    ),
    (
        ["cognitive", "reasoning", "aptitude", "ability"],
        ["ability", "reasoning", "aptitude"],
        1.5
    ),
    (
        ["graduate", "entry level", "fresher", "intern"],
        ["graduate", "entry"],
        1.2
    ),
    (
        ["developer", "engineer", "software"],
        ["developer", "engineer", "software"],
        1.3
    ),
]


class HybridRetriever:

    def __init__(self, catalog):

        self.catalog = catalog

        self.documents = []

        for item in catalog:

            text = " ".join([
                item.get("name", ""),
                item.get("description", ""),
                " ".join(item.get("keys", [])),
                " ".join(item.get("job_levels", [])),
                " ".join(item.get("languages", []))
            ]).lower()

            self.documents.append(text.split())

        self.bm25 = BM25Okapi(self.documents)

    def _domain_boost(self, query, item):

        query_lower = query.lower()

        combined = " ".join([
            item.get("name", ""),
            item.get("description", ""),
            " ".join(item.get("keys", [])),
            " ".join(item.get("job_levels", [])),
            " ".join(item.get("languages", []))
        ]).lower()

        score = 0.0

        for triggers, signals, bonus in DOMAIN_BOOSTS:

            if any(trigger in query_lower for trigger in triggers):

                if any(signal in combined for signal in signals):

                    score += bonus

        query_words = set(query_lower.split())

        overlap = sum(
            1 for word in query_words
            if len(word) > 3 and word in combined
        )

        score += overlap * 0.05

        return score

    def search(
        self,
        query,
        conversation_context="",
        top_k=10
    ):

        full_query = f"{query} {conversation_context}".strip()

        tokens = full_query.lower().split()

        scores = self.bm25.get_scores(tokens)

        ranked = []

        for idx, item in enumerate(self.catalog):

            score = float(scores[idx])

            score += self._domain_boost(full_query, item)

            ranked.append((score, item))

        ranked.sort(
            key=lambda x: x[0],
            reverse=True
        )

        return [
            item
            for score, item in ranked[:top_k]
        ]
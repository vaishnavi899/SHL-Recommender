"""
retriever.py -- Hybrid semantic + BM25 + domain-boost retriever for the SHL catalog.

Key design decisions
--------------------
* Semantic index (FAISS / all-MiniLM-L6-v2) captures meaning.
* BM25 uses DEDUPLICATED tokens from the conversation context to prevent
  noise words like "compare", "yes", "selection" (repeated across many turns)
  from inflating BM25 scores and surfacing irrelevant items.
* Semantic scoring blends the primary query (70%) with the full context (30%)
  so accumulated constraints carry through without losing precision.
* Domain boost table is data-driven: add new domains without touching scoring.
* Seniority penalty demotes items whose job_levels clearly conflict with the
  seniority implied in the query (e.g. entry-level tests for CXO search).
* BM25 and semantic scores are min-max normalised before fusion.
"""

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

DOMAIN_BOOSTS = [
    (["leadership", "leader"],
     ["leadership", "executive"],                          0.45),
    (["manager", "management"],
     ["manager", "management", "leadership"],              0.40),
    (["executive", "c-suite", "vp", "cxo", "director"],
     ["executive", "leadership", "director"],              0.50),
    (["personality", "behaviour", "behavioral", "behavioural"],
     ["personality", "behaviour"],                         0.40),
    (["cognitive", "reasoning", "aptitude", "ability"],
     ["cognitive", "reasoning", "numerical",
      "verbal", "inductive", "deductive"],                 0.40),
    (["sales", "selling"],
     ["sales"],                                            0.50),
    (["customer", "service", "support"],
     ["customer", "service"],                              0.35),
    (["java", "jvm"],
     ["java"],                                             0.55),
    (["python"],
     ["python"],                                           0.55),
    (["javascript", "js", "node", "react", "frontend"],
     ["javascript", "js"],                                 0.50),
    (["sql", "database", "data"],
     ["sql", "database", "data"],                          0.45),
    (["developer", "engineer", "programmer", "software", "coding"],
     ["developer", "engineer", "software", "programming"], 0.35),
    (["communication", "written", "verbal", "stakeholder"],
     ["communication", "interpersonal", "verbal", "written"], 0.35),
    (["numerical", "finance", "accounting", "numbers"],
     ["numerical", "finance", "accounting"],               0.40),
    (["verbal", "reading", "comprehension"],
     ["verbal", "reading", "comprehension"],               0.35),
    (["graduate", "entry level", "fresh", "trainee"],
     ["graduate", "entry"],                                0.30),
    (["senior", "experienced", "mid-level", "mid level"],
     ["senior", "professional", "mid"],                    0.25),
    (["call centre", "call center", "contact centre"],
     ["call centre", "contact centre", "customer service"], 0.45),
    (["biodata", "background"],
     ["biodata"],                                          0.45),
    (["situational judgement", "sjt"],
     ["situational", "judgement"],                         0.45),
    (["motivation", "values", "culture fit"],
     ["motivation", "values", "culture"],                  0.35),
    (["agile", "scrum"],
     ["agile", "scrum", "software"],                       0.30),
    (["data science", "ml", "machine learning", "ai"],
     ["data", "analytics", "python", "sql"],               0.35),
    (["nursing", "clinical", "healthcare", "medical"],
     ["nursing", "clinical", "healthcare"],                0.45),
    (["hipo", "high potential", "high-potential"],
     ["hipo", "potential", "aspiration"],                  0.50),
    (["benchmark", "benchmarking"],
     ["benchmark", "leadership", "enterprise"],            0.45),
]

_SENIOR_QUERY_SIGNALS = {
    "executive", "director", "cxo", "senior", "leadership",
    "c-suite", "15 years", "15+ years", "vp", "vice president",
}
_JUNIOR_QUERY_SIGNALS = {
    "graduate", "entry", "junior", "trainee", "fresher", "intern", "campus",
}
_SENIOR_CONTENT_SIGNALS = {
    "director", "executive", "senior", "manager", "leadership",
    "professional", "mid-professional",
}
_JUNIOR_CONTENT_SIGNALS = {"entry", "graduate", "trainee", "junior"}


class HybridRetriever:

    def __init__(self, catalog: list):
        self.catalog = catalog

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.documents = []
        for item in catalog:
            text = " ".join(filter(None, [
                item.get("name", ""),
                item.get("description", ""),
                " ".join(item.get("keys", [])),
                " ".join(item.get("job_levels", [])),
                " ".join(item.get("languages", [])),
            ]))
            self.documents.append(text)

        self.tokenized_docs = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

        embeddings = self.embedding_model.encode(
            self.documents, convert_to_numpy=True, show_progress_bar=False
        )
        self.embeddings = embeddings.astype("float32")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)


    def _domain_boost(self, query: str, item: dict) -> float:
        """Data-driven domain boost using the DOMAIN_BOOSTS table."""
        query_lower = query.lower()
        name = item.get("name", "").lower()
        desc = item.get("description", "").lower()
        keys = " ".join(item.get("keys", [])).lower()
        combined = f"{name} {desc} {keys}"

        score = 0.0
        for query_triggers, content_signals, bonus in DOMAIN_BOOSTS:
            if any(t in query_lower for t in query_triggers):
                if any(s in combined for s in content_signals):
                    score += bonus

        query_words = set(w for w in query_lower.split() if len(w) > 3)
        overlap = sum(1 for w in query_words if w in combined)
        score += overlap * 0.04

        return score

    def _seniority_penalty(self, query: str, item: dict) -> float:
        """
        Return a negative score when the retrieved item's job levels clearly
        conflict with the seniority implied by the query.

        E.g. a query about CXOs/directors should penalise entry-level tests
        like Visual Comparison or call-centre assessments.
        """
        query_lower = query.lower()
        job_levels_text = " ".join(item.get("job_levels", [])).lower()

        wants_senior = any(s in query_lower for s in _SENIOR_QUERY_SIGNALS)
        wants_junior = any(s in query_lower for s in _JUNIOR_QUERY_SIGNALS)

        if wants_senior:
            item_is_junior_only = (
                any(s in job_levels_text for s in _JUNIOR_CONTENT_SIGNALS)
                and not any(s in job_levels_text for s in _SENIOR_CONTENT_SIGNALS)
            )
            if item_is_junior_only:
                return -0.40

        if wants_junior:
            item_is_senior_only = (
                any(s in job_levels_text for s in _SENIOR_CONTENT_SIGNALS)
                and not any(s in job_levels_text for s in _JUNIOR_CONTENT_SIGNALS)
            )
            if item_is_senior_only:
                return -0.30

        return 0.0

    @staticmethod
    def _normalise(scores: np.ndarray) -> np.ndarray:
        """Min-max normalise to [0, 1]. Returns zeros if all scores are equal."""
        lo, hi = scores.min(), scores.max()
        if hi - lo < 1e-9:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)


    def search(
        self,
        query: str,
        conversation_context: str = "",
        top_k: int = 10,
    ) -> list:
        """
        Retrieve the top-k catalog items for a query.

        Parameters
        ----------
        query                : The latest user message (primary signal).
        conversation_context : All user turns joined. Deduplicated before
                               BM25 use to prevent noise-word inflation.
        top_k                : Number of results to return.
        """
        n = len(self.catalog)

        q_emb = self.embedding_model.encode(
            [query], convert_to_numpy=True
        ).astype("float32")
        distances, indices = self.index.search(q_emb, min(top_k * 3, n))

        raw_semantic = np.zeros(n, dtype="float32")
        for rank, idx in enumerate(indices[0]):
            raw_semantic[idx] = 1.0 / (1.0 + distances[0][rank])
        primary_semantic = self._normalise(raw_semantic)

        has_context = (
            bool(conversation_context)
            and conversation_context.strip() != query.strip()
        )
        if has_context:
            ctx_emb = self.embedding_model.encode(
                [conversation_context], convert_to_numpy=True
            ).astype("float32")
            ctx_dist, ctx_idx = self.index.search(ctx_emb, min(top_k * 3, n))

            raw_ctx = np.zeros(n, dtype="float32")
            for rank, idx in enumerate(ctx_idx[0]):
                raw_ctx[idx] = 1.0 / (1.0 + ctx_dist[0][rank])
            context_semantic = self._normalise(raw_ctx)

            blended_semantic = 0.70 * primary_semantic + 0.30 * context_semantic
        else:
            blended_semantic = primary_semantic

        if has_context:
            raw_tokens = f"{query} {conversation_context}".lower().split()
            seen: set = set()
            bm25_tokens = []
            for tok in raw_tokens:
                if tok not in seen:
                    seen.add(tok)
                    bm25_tokens.append(tok)
        else:
            bm25_tokens = query.lower().split()

        raw_bm25 = np.array(
            self.bm25.get_scores(bm25_tokens), dtype="float32"
        )
        bm25_scores = self._normalise(raw_bm25)

        boost_query = (
            f"{query} {conversation_context}" if has_context else query
        )

        results = []
        for idx in range(n):
            item = self.catalog[idx]
            domain_boost = self._domain_boost(boost_query, item)
            seniority_pen = self._seniority_penalty(boost_query, item)

            final = (
                0.55 * blended_semantic[idx]
                + 0.30 * bm25_scores[idx]
                + domain_boost
                + seniority_pen
            )
            results.append((final, item))

        results.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in results[:top_k]]
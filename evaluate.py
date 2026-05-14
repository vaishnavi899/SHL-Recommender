import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Optional

import requests


TEST_TRACES = [
    {
        "id": "senior_leadership",
        "description": "CXO/director hiring with personality + cognitive + leadership style",
        "conversation": [
            {"role": "user", "content": "We need a solution for senior leadership hiring"},
            {"role": "assistant", "content": "Could you tell me the target level and what you need to assess?"},
            {"role": "user", "content": "CXOs and directors with 15+ years experience"},
            {"role": "assistant", "content": "Is this for selection or development?"},
            {"role": "user", "content": "Selection against leadership benchmark with personality and cognitive evaluation"},
            {"role": "assistant", "content": "Would you also like leadership style reporting?"},
            {"role": "user", "content": "Yes, and compare the best options"},
        ],
        "expected_names": [
            "Enterprise Leadership Report",
            "OPQ Leadership Report",
            "HiPo Assessment Report",
            "Executive Scenarios",
        ],
        "forbidden_names": [
            "Visual Comparison",
            "Cisco AppDynamics",
            "Econometrics",
        ],
        "must_clarify_before_turn": 3,
    },
    {
        "id": "java_developer",
        "description": "Mid-level Java developer with stakeholder communication needs",
        "conversation": [
            {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
            {"role": "assistant", "content": "What seniority level are you hiring for?"},
            {"role": "user", "content": "Mid-level, around 4 years experience"},
        ],
        "expected_names": [
            "Java",
        ],
        "forbidden_names": [
            "Visual Comparison",
            "Executive Scenarios",
        ],
        "must_clarify_before_turn": 2,
    },
    {
        "id": "vague_start",
        "description": "Agent must clarify before recommending on a vague first message",
        "conversation": [
            {"role": "user", "content": "I need an assessment"},
        ],
        "expected_names": [],
        "forbidden_names": [],
        "must_clarify_before_turn": 1,
        "expect_empty_recommendations": True,
    },
    {
        "id": "sales_role",
        "description": "Entry-level sales representative",
        "conversation": [
            {"role": "user", "content": "Looking to hire entry-level sales reps for a call centre"},
            {"role": "assistant", "content": "Are you assessing personality, ability, or both?"},
            {"role": "user", "content": "Both personality and verbal reasoning"},
        ],
        "expected_names": [
            "Sales",
            "Customer Service",
            "Verbal",
        ],
        "forbidden_names": [
            "Executive",
            "Enterprise Leadership",
        ],
        "must_clarify_before_turn": 2,
    },
]


@dataclass
class MetricResult:
    name: str
    score: float
    passed: bool
    detail: str


@dataclass
class TraceResult:
    trace_id: str
    description: str
    metrics: list[MetricResult] = field(default_factory=list)
    raw_response: Optional[dict] = None
    error: Optional[str] = None

    @property
    def overall_score(self) -> float:
        if not self.metrics:
            return 0.0
        return sum(m.score for m in self.metrics) / len(self.metrics)


def evaluate_retrieval(catalog: list, top_k: int = 10):

    from app.retriever import HybridRetriever

    retriever = HybridRetriever(catalog)

    retrieval_cases = [
        {
            "query": "senior executive leadership personality cognitive benchmark",
            "context": "CXOs directors 15 years selection leadership benchmark personality cognitive",
            "expected": ["leadership", "enterprise", "hipo", "executive", "opq"],
            "forbidden": ["visual comparison", "cisco", "econometrics"],
        },
        {
            "query": "java developer mid level",
            "context": "java developer stakeholder communication",
            "expected": ["java"],
            "forbidden": ["executive scenarios", "enterprise leadership"],
        },
    ]

    precision_scores = []
    recall_scores = []
    mrr_scores = []

    for case in retrieval_cases:

        results = retriever.search(
            query=case["query"],
            conversation_context=case["context"],
            top_k=top_k,
        )

        names = [r.get("name", "").lower() for r in results]

        relevant_hits = sum(
            1 for name in names
            if any(exp in name for exp in case["expected"])
        )

        precision_scores.append(relevant_hits / top_k)

        recall = sum(
            1 for exp in case["expected"]
            if any(exp in name for name in names)
        ) / len(case["expected"])

        recall_scores.append(recall)

        rr = 0.0

        for rank, name in enumerate(names, start=1):

            if any(exp in name for exp in case["expected"]):
                rr = 1.0 / rank
                break

        mrr_scores.append(rr)

    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_mrr = sum(mrr_scores) / len(mrr_scores)

    return [
        MetricResult(
            name="retrieval_precision_at_k",
            score=round(avg_precision, 3),
            passed=avg_precision >= 0.30,
            detail=f"Precision@{top_k}: {avg_precision:.2f}",
        ),
        MetricResult(
            name="retrieval_recall_at_k",
            score=round(avg_recall, 3),
            passed=avg_recall >= 0.60,
            detail=f"Recall@{top_k}: {avg_recall:.2f}",
        ),
        MetricResult(
            name="retrieval_mrr",
            score=round(avg_mrr, 3),
            passed=avg_mrr >= 0.50,
            detail=f"MRR: {avg_mrr:.2f}",
        ),
    ]


def score_recommendation_relevance(
    response,
    expected_names,
    forbidden_names,
    expect_empty=False,
):

    recs = response.get("recommendations", [])

    rec_names = [r.get("name", "").lower() for r in recs]

    metrics = []

    if expect_empty:

        passed = len(recs) == 0

        metrics.append(
            MetricResult(
                name="empty_recommendations",
                score=1.0 if passed else 0.0,
                passed=passed,
                detail=f"{len(recs)} recommendations returned",
            )
        )

        return metrics

    hits = sum(
        1 for exp in expected_names
        if any(exp.lower() in name for name in rec_names)
    )

    hit_rate = hits / max(len(expected_names), 1)

    metrics.append(
        MetricResult(
            name="expected_recommendation_hit_rate",
            score=round(hit_rate, 3),
            passed=hit_rate >= 0.50,
            detail=f"{hits}/{len(expected_names)} expected categories matched",
        )
    )

    forbidden_hits = sum(
        1 for forb in forbidden_names
        if any(forb.lower() in name for name in rec_names)
    )

    forbidden_score = 1.0 - (
        forbidden_hits / max(len(forbidden_names), 1)
    )

    metrics.append(
        MetricResult(
            name="forbidden_recommendations_absent",
            score=round(forbidden_score, 3),
            passed=forbidden_score == 1.0,
            detail=f"{forbidden_hits} forbidden matches",
        )
    )

    return metrics


def score_groundedness(response, catalog):

    reply = response.get("reply", "").lower()

    rec_names = {
        r.get("name", "").lower()
        for r in response.get("recommendations", [])
    }

    catalog_names = {
        item.get("name", "").lower()
        for item in catalog
    }

    mentioned = [
        name for name in catalog_names
        if len(name) > 8 and name in reply
    ]

    if not mentioned:

        return [
            MetricResult(
                name="groundedness",
                score=1.0,
                passed=True,
                detail="No named products in reply",
            )
        ]

    valid = sum(
        1 for name in mentioned
        if name in rec_names
    )

    score = valid / len(mentioned)

    return [
        MetricResult(
            name="groundedness",
            score=round(score, 3),
            passed=score >= 0.80,
            detail=f"{valid}/{len(mentioned)} reply products grounded",
        )
    ]


def score_conversation_behavior(response, trace):

    metrics = []

    reply = response.get("reply", "")
    recs = response.get("recommendations", [])
    eoc = response.get("end_of_conversation", False)

    metrics.append(
        MetricResult(
            name="non_empty_reply",
            score=1.0 if reply.strip() else 0.0,
            passed=bool(reply.strip()),
            detail=f"Reply length {len(reply)}",
        )
    )

    if recs:
        metrics.append(
            MetricResult(
                name="end_of_conversation",
                score=1.0 if eoc else 0.0,
                passed=eoc,
                detail=f"end_of_conversation={eoc}",
            )
        )

    return metrics


def run_trace(endpoint, trace, catalog):

    result = TraceResult(
        trace_id=trace["id"],
        description=trace["description"],
    )

    try:

        start = time.time()

        response = requests.post(
            f"{endpoint}/chat",
            json={"messages": trace["conversation"]},
            timeout=28,
        )

        latency = time.time() - start

        if response.status_code != 200:
            result.error = f"HTTP {response.status_code}"
            return result

        data = response.json()

        result.raw_response = data

        result.metrics.append(
            MetricResult(
                name="latency",
                score=max(0.0, 1.0 - latency / 30),
                passed=latency < 30,
                detail=f"{latency:.2f}s",
            )
        )

        result.metrics.extend(
            score_recommendation_relevance(
                data,
                trace.get("expected_names", []),
                trace.get("forbidden_names", []),
                trace.get("expect_empty_recommendations", False),
            )
        )

        result.metrics.extend(
            score_groundedness(data, catalog)
        )

        result.metrics.extend(
            score_conversation_behavior(data, trace)
        )

    except Exception as e:
        result.error = str(e)

    return result


def print_metric(metric):

    status = "PASS" if metric.passed else "FAIL"

    print(
        f"{status} | "
        f"{metric.name} | "
        f"{metric.score:.3f} | "
        f"{metric.detail}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000",
    )

    parser.add_argument(
        "--catalog",
        default="data/shl_product_catalog.json",
    )

    args = parser.parse_args()

    with open(args.catalog, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    print("\n=== RETRIEVAL EVALUATION ===\n")

    retrieval_metrics = evaluate_retrieval(catalog)

    for metric in retrieval_metrics:
        print_metric(metric)

    print("\n=== LIVE API EVALUATION ===\n")

    trace_results = []

    for trace in TEST_TRACES:

        result = run_trace(
            args.endpoint,
            trace,
            catalog,
        )

        trace_results.append(result)

        print(f"\nTRACE: {trace['id']}")

        if result.error:
            print(f"ERROR: {result.error}")
            continue

        for metric in result.metrics:
            print_metric(metric)

        print(
            f"OVERALL SCORE: "
            f"{result.overall_score:.3f}"
        )

    overall_scores = []

    for metric in retrieval_metrics:
        overall_scores.append(metric.score)

    for result in trace_results:
        for metric in result.metrics:
            overall_scores.append(metric.score)

    final_score = (
        sum(overall_scores) / len(overall_scores)
        if overall_scores else 0.0
    )

    print("\n==============================")
    print(f"FINAL SCORE: {final_score:.3f}")
    print("==============================\n")


if __name__ == "__main__":
    main()
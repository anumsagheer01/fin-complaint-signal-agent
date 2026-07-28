"""
Evaluates retrieval quality: for held-out complaint narratives, checks whether
the retriever surfaces documents with the same issue label (a proxy for
'did it find genuinely related complaints').
"""
import random
import csv
from pathlib import Path
from retrieval import load_index, retrieve

random.seed(42)

RESULTS_PATH = Path(__file__).parent.parent / "data" / "processed" / "retrieval_eval_results.csv"


def build_query_from_narrative(narrative, max_words=25):
    # Simulate a realistic user query: first ~25 words of a real complaint,
    # framed as a question, rather than pasting the whole narrative.
    words = narrative.split()[:max_words]
    return " ".join(words)


def run_eval(n_samples=200, k=5):
    index = load_index()
    docs = index["docs"]

    sample = random.sample(docs, n_samples)

    hits_at_k = 0
    hits_at_1 = 0
    category_match_at_k = 0
    rows = []

    for doc in sample:
        query = build_query_from_narrative(doc["narrative"])
        results = retrieve(query, index, k=k)

        retrieved_ids = [r["doc_id"] for r in results]
        retrieved_issues = [r["issue"] for r in results]
        retrieved_categories = [r["category"] for r in results]

        self_found = doc["doc_id"] in retrieved_ids
        issue_hit = doc["issue"] in retrieved_issues
        category_hit = doc["category"] in retrieved_categories
        top1_issue_match = results[0]["issue"] == doc["issue"] if results else False

        if issue_hit:
            hits_at_k += 1
        if top1_issue_match:
            hits_at_1 += 1
        if category_hit:
            category_match_at_k += 1

        rows.append({
            "query_doc_id": doc["doc_id"],
            "true_issue": doc["issue"],
            "true_category": doc["category"],
            "self_retrieved": self_found,
            "issue_match_at_k": issue_hit,
            "category_match_at_k": category_hit,
            "top1_issue_match": top1_issue_match,
        })

    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"=== Retrieval Eval Results (n={n_samples}, k={k}) ===")
    print(f"Self-document retrieval rate (sanity check, should be ~100%): {sum(r['self_retrieved'] for r in rows)/n_samples:.1%}")
    print(f"Issue label match @ top-1: {hits_at_1/n_samples:.1%}")
    print(f"Issue label match @ top-{k}: {hits_at_k/n_samples:.1%}")
    print(f"Category label match @ top-{k}: {category_match_at_k/n_samples:.1%}")
    print(f"Results saved to {RESULTS_PATH}")

    return {
        "n_samples": n_samples,
        "k": k,
        "self_retrieval_rate": sum(r['self_retrieved'] for r in rows)/n_samples,
        "issue_match_at_1": hits_at_1/n_samples,
        "issue_match_at_k": hits_at_k/n_samples,
        "category_match_at_k": category_match_at_k/n_samples,
    }


if __name__ == "__main__":
    run_eval(n_samples=200, k=5)

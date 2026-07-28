"""
TF-IDF based document retrieval over the complaint narrative corpus.
This is the retrieval half of the RAG pipeline.
"""
import csv
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "rag_corpus.csv"
INDEX_PATH = Path(__file__).parent.parent / "data" / "processed" / "retrieval_index.pkl"


def load_corpus():
    docs = []
    with open(DATA_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            docs.append(row)
    return docs


def build_index(docs):
    texts = [d["narrative"] for d in docs]
    vectorizer = TfidfVectorizer(
        max_features=20000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def save_index(vectorizer, matrix, docs):
    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": matrix, "docs": docs}, f)


def load_index():
    with open(INDEX_PATH, "rb") as f:
        return pickle.load(f)


def retrieve(query, index, k=5):
    q_vec = index["vectorizer"].transform([query])
    sims = cosine_similarity(q_vec, index["matrix"]).flatten()
    top_k_idx = sims.argsort()[::-1][:k]
    results = []
    for i in top_k_idx:
        doc = index["docs"][i]
        results.append({
            "doc_id": doc["doc_id"],
            "category": doc["category"],
            "issue": doc["issue"],
            "score": float(sims[i]),
            "narrative_snippet": doc["narrative"][:200],
        })
    return results


if __name__ == "__main__":
    print("Loading corpus...")
    docs = load_corpus()
    print(f"Loaded {len(docs)} documents")

    print("Building TF-IDF index...")
    vectorizer, matrix = build_index(docs)
    print(f"Index shape: {matrix.shape}")

    save_index(vectorizer, matrix, docs)
    print(f"Saved index to {INDEX_PATH}")

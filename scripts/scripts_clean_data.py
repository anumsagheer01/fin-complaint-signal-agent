import csv
import re
from collections import Counter, defaultdict

RAW_DIR = "/home/claude/project/data_raw"
OUT_DIR = "/home/claude/project/data_processed"

FILES = ["bank_of_america_complaints.csv", "jpmorgan_complaints.csv"]

TARGET_CATEGORIES = {
    "Checking or savings account",
    "Money transfer, virtual currency, or money service",
}

# Data pull cuts off mid-week at the end (2025-10-01 through 2025-10-08 show
# an artificial drop to single-digit daily counts vs. a normal 30-80/day range).
# That's a truncated pull window, not a real anomaly, so we exclude it.
END_DATE = "2025-09-30"

all_rows = []
for fname in FILES:
    with open(f"{RAW_DIR}/{fname}") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["product"] not in TARGET_CATEGORIES:
                continue
            if row["date_received"][:10] > END_DATE:
                continue
            all_rows.append(row)

print(f"Total rows across both institutions, target categories only: {len(all_rows)}")

cat_counts = Counter(r["product"] for r in all_rows)
print("By category:", dict(cat_counts))

inst_counts = Counter(r["company"] for r in all_rows)
print("By institution:", dict(inst_counts))

# ---- 1. Daily volume time series per category (institutions combined) ----
daily_counts = defaultdict(lambda: defaultdict(int))
for r in all_rows:
    date = r["date_received"][:10]
    if not date:
        continue
    daily_counts[date][r["product"]] += 1

dates_sorted = sorted(daily_counts.keys())
print(f"Date range: {dates_sorted[0]} to {dates_sorted[-1]}")

with open(f"{OUT_DIR}/daily_volume_timeseries.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["date", "category", "complaint_count"])
    for date in dates_sorted:
        for cat in TARGET_CATEGORIES:
            writer.writerow([date, cat, daily_counts[date].get(cat, 0)])

print("Wrote daily_volume_timeseries.csv")

# ---- 1b. Daily volume time series per category, split by institution (for later comparative cuts) ----
daily_counts_by_inst = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for r in all_rows:
    date = r["date_received"][:10]
    if not date:
        continue
    daily_counts_by_inst[date][r["company"]][r["product"]] += 1

with open(f"{OUT_DIR}/daily_volume_timeseries_by_institution.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["date", "institution", "category", "complaint_count"])
    institutions = sorted(inst_counts.keys())
    for date in dates_sorted:
        for inst in institutions:
            for cat in TARGET_CATEGORIES:
                writer.writerow([date, inst, cat, daily_counts_by_inst[date][inst].get(cat, 0)])

print("Wrote daily_volume_timeseries_by_institution.csv")

# ---- 2. RAG corpus: rows with real narrative text ----
narrative_rows = [r for r in all_rows if r["complaint_what_happened"].strip()]
print(f"Rows with narrative text: {len(narrative_rows)}")

def clean_narrative(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text

with open(f"{OUT_DIR}/rag_corpus.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["doc_id", "institution", "category", "sub_issue", "issue", "date_received", "narrative"])
    doc_id = 0
    for r in narrative_rows:
        narrative = clean_narrative(r["complaint_what_happened"])
        if len(narrative) < 40:
            continue
        doc_id += 1
        writer.writerow([
            f"doc_{doc_id:06d}",
            r["company"],
            r["product"],
            r.get("sub_issue", ""),
            r.get("issue", ""),
            r["date_received"][:10],
            narrative,
        ])

print(f"Wrote rag_corpus.csv with {doc_id} documents")

# ---- 3. Combined clean metadata file ----
with open(f"{OUT_DIR}/complaints_clean.csv", "w", newline="") as f:
    fieldnames = ["complaint_id", "institution", "product", "sub_product", "issue", "sub_issue",
                  "date_received", "date_sent_to_company", "state", "timely",
                  "company_response", "submitted_via", "has_narrative"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in all_rows:
        writer.writerow({
            "complaint_id": r["complaint_id"],
            "institution": r["company"],
            "product": r["product"],
            "sub_product": r.get("sub_product", ""),
            "issue": r.get("issue", ""),
            "sub_issue": r.get("sub_issue", ""),
            "date_received": r["date_received"][:10],
            "date_sent_to_company": r.get("date_sent_to_company", "")[:10],
            "state": r.get("state", ""),
            "timely": r.get("timely", ""),
            "company_response": r.get("company_response", ""),
            "submitted_via": r.get("submitted_via", ""),
            "has_narrative": bool(r["complaint_what_happened"].strip()),
        })

print("Wrote complaints_clean.csv")

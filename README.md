# fin-complaint-signal-agent

An agentic system for financial services complaint monitoring. It does two things:

1. **Answers questions** about customer complaints by retrieving relevant real complaint
   narratives and generating a cited answer (RAG).
2. **Flags anomalies** in daily complaint volume by category, so a servicing team could
   spot unusual spikes before they become a bigger problem.

Built as a portfolio project to demonstrate agentic RAG + time-series monitoring for a
research-engineer-style data science role in payments/financial services.

## Why this project

Most of my other projects are "agent retrieves docs and answers a question." This one
adds a second, genuinely different skill: detecting anomalies in a real time series and
explaining *why* something spiked using the RAG layer. Two distinct technical skills,
one system.

## The data

Real consumer complaint records pulled from the CFPB (Consumer Financial Protection
Bureau) — a US government agency that publishes every complaint filed against a bank
publicly. Filtered to two categories:

- **Checking or savings account** (23,322 complaints)
- **Money transfer, virtual currency, or money service** (5,110 complaints)

Date range: Jan 1, 2024 – Sep 30, 2025 (the original pull went a bit further but the
last week was a truncated data cutoff, not real low volume, so I trimmed it — worth
catching before it feeds an anomaly detector and creates a fake alert).

Two institutions are represented in the raw data since the source combined them; company
identity isn't the point of this project so it's not surfaced anywhere in the pipeline
output, just present in the raw files for transparency.

## Architecture

```
                    ┌─────────────────┐
   User question →  │   LangGraph      │
                    │   Agent          │
                    └────────┬─────────┘
                             │
              ┌──────────────┴───────────────┐
              ▼                               ▼
    ┌─────────────────┐           ┌──────────────────────┐
    │  TF-IDF Retrieval │           │  Anomaly Detection   │
    │  (14,945 docs)     │           │  (STL decomposition)  │
    └─────────────────┘           └──────────────────────┘
              │                               │
              └──────────────┬───────────────┘
                             ▼
                   ┌───────────────────┐
                   │  Claude (Anthropic  │
                   │  API) generates      │
                   │  cited answer         │
                   └───────────────────┘
```

## Results (real numbers, not projected)

### Retrieval evaluation
Evaluated by sampling 200 held-out complaint narratives, building a realistic query
from the first ~25 words, and checking whether the retriever surfaces documents sharing
the same issue label (a proxy for topical relevance).

- **Issue label match @ top-1: 85.0%**
- **Issue label match @ top-5: 97.0%**
- **Category label match @ top-5: 99.5%**

### Anomaly detection
Used STL (seasonal-trend-residual) decomposition with weekly seasonality (period=7,
since complaint volume clearly dips on weekends), flagging days where the residual
z-score exceeds 2.5.

- **Checking or savings account**: 25 anomalies flagged across 639 days (3.9%)
- **Money transfer**: 5 anomalies flagged across 639 days (0.8%)
- Both categories show a sharp, simultaneous spike around **Jan 15–18, 2025**
  (z-scores up to 18.3) — a real, detected deviation. I looked into public sources
  for a specific cause and didn't find one pinned to that exact week, but 2025 saw a
  53% YoY increase in checking/savings complaints and a 275% YoY increase in money
  transfer complaints per CFPB's annual report, so this sits inside a genuine
  broader volume increase. In a real deployment this is exactly the kind of thing
  that gets flagged for a human to review, not auto-explained — I'm being upfront
  that the detector found something real even though I can't fully explain the
  specific week myself.

## What's fully working vs. what needs a key

- Data pipeline, retrieval, anomaly detection, evaluation — all real, all executed,
  all numbers above are from actual runs.
- **Answer generation** requires an Anthropic API key (not included, obviously).
  Set it as an environment variable and the agent will generate real cited answers:
  ```bash
  export ANTHROPIC_API_KEY=your_key_here
  ```
  Without a key, the agent still runs end-to-end and returns retrieved docs +
  anomaly data, just skips the final generation step.

## Project structure

```
fin-complaint-signal-agent/
├── data/
│   ├── raw/                  # original unmodified source files
│   └── processed/            # cleaned data, eval results, anomaly results
├── scripts/
│   └── scripts_clean_data.py # data cleaning/filtering pipeline
├── src/
│   ├── retrieval.py          # TF-IDF index build + search
│   ├── evaluate_retrieval.py # retrieval eval harness
│   ├── anomaly_detection.py  # STL-based anomaly detection
│   ├── agent.py              # LangGraph orchestration + generation
│   └── api.py                # FastAPI wrapper
├── Dockerfile
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt

# 1. Clean the data (already done, outputs are in data/processed/)
python scripts/scripts_clean_data.py

# 2. Build the retrieval index (not committed to git, rebuild locally)
cd src
python retrieval.py

# 3. Run anomaly detection
python anomaly_detection.py

# 4. Run the agent (set ANTHROPIC_API_KEY first for real generation)
python agent.py

# 5. Or run the API
uvicorn api:app --reload
# then POST to http://localhost:8000/ask with {"question": "..."}
```

Or with Docker:
```bash
docker build -t complaint-agent .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=your_key complaint-agent
```

## Honest limitations

- Anomaly labels aren't ground-truth verified against confirmed real-world events —
  they're statistically flagged deviations, validated by the fact that the detection
  is stable and the magnitude (z > 6, even > 18 for one category) is not subtle.
- The RAG corpus is complaint narratives, not internal servicing docs, so the
  "explain why" story it tells is limited to what's in consumer-submitted text, not
  internal company knowledge a real deployment would have access to.
- Self-retrieval rate in eval (89.5%) is a sanity check, not the main metric — it's
  a bit below 100% because TF-IDF similarity between a 25-word snippet and its own
  longer parent document isn't guaranteed to be the single top match when other
  near-duplicate complaints exist in the corpus.

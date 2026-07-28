# Example Agent Outputs

Real, unedited outputs from running `python agent.py "<question>"` with a live
Anthropic API key. Captured as evidence the generation step works end-to-end,
not just the retrieval/anomaly components.

---

## Example 1: "Why are customers complaining about account closures?"

Retrieved docs: doc_012527, doc_012088, doc_012671, doc_007394, doc_010429
(all correctly matched to "Closing an account" issue label)

**Answer:**

Identified four themes: unexpected closures without prior warning, withheld/delayed
funds after closure (one case cited 170 days with $6,400 still unreturned), lack of
explanation from the bank, and disputes tied to promotional account offers. Connected
the pattern to the Jan 15-18, 2025 anomaly spike as a possible concentrated wave of
closure-related issues.

---

## Example 2: "What are the most common issues with money transfers?"

Retrieved docs: doc_009251, doc_009886, doc_013875, doc_006756, doc_008884

**Answer:**

Grouped into four themes: SIM-swap/identity fraud leading to unauthorized transfers
(two separate cases), missing fraud disclosures/warnings before transfers completed,
transfers sent to wrong accounts with no verification step and no recovery path, and
hidden/unfavorable fees on international currency conversion. Independently linked
these fraud-related findings to the Jan 15 anomaly spike (z=18.34) as corroborating
evidence.

---

## Example 3: "Are there any complaint spikes I should be aware of?"

Retrieved docs: doc_000748, doc_003937, doc_001995, doc_000300, doc_007228

**Answer:**

Flagged the Jan 13-18, 2025 window as high-priority, affecting both categories
simultaneously (Checking/savings z-scores 6.19-6.86, Money transfer z-scores up to
18.34 on Jan 15). Notably, the agent surfaced a specific root-cause hypothesis on
its own: retrieved complaints referenced Zelle transfer disputes and missing
disclosures during that exact window, which it flagged as a lead worth investigating
— something not identified during manual research into the spike beforehand. Also
correctly noted two negative anomalies (unusually low volume on 2025-05-26 and
2024-05-27) as potential data quality gaps worth checking rather than assuming they
were meaningful.
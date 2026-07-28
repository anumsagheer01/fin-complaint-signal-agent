# fin-complaint-signal-agent

This project is an AI agent that does two things with real customer complaint data:

1. It answers questions about complaints by pulling up the most relevant real complaints and writing a summary, with sources cited.
2. It watches complaint volume over time and flags days where something unusual is happening, like a sudden spike.

I built this as a portfolio project to show two skills together: retrieval based AI agents (RAG) and time series anomaly detection. 

## What data does it use

Real complaint data from the CFPB, a US government agency that publishes every complaint filed against a bank. I filtered it down to two categories:

- Checking or savings account complaints (23,322 of them)
- Money transfer and virtual currency complaints (5,110 of them)

The data covers January 2024 through September 2025. My original pull went a little further than that, but the last week of data was cut off mid pull, which made it look like complaints suddenly dropped to almost zero. That was not real, it was just an incomplete data pull, so I trimmed it out. Catching that kind of thing matters, because feeding fake low numbers into an anomaly detector would create a false alarm.

## How it works

A user asks a question. The agent does two things at once:

1. It searches through 14,945 real complaint documents to find the most relevant ones.
2. It checks whether there have been any recent unusual spikes in complaint volume.

Then it hands both of those to Claude, which writes an answer that cites the specific complaints it used.

```
Question
   |
   v
Agent
   |
   +----> Search complaints (TF-IDF retrieval)
   |
   +----> Check for volume spikes (anomaly detection)
   |
   v
Claude writes a cited answer
```

## Results, from actually running it

### How good is the search

I tested this by taking 200 real complaints, turning the first few sentences of each into a search query, and checking whether the system found other complaints about the same issue.

- It got the exact right issue type in its top result 85 percent of the time
- It got the right issue type somewhere in its top 5 results 97 percent of the time
- It got the right general category in its top 5 results 99.5 percent of the time

### What the anomaly detector found

I used a method called STL decomposition, which separates a time series into trend, weekly pattern, and leftover noise, then flags days where that leftover noise is much bigger than normal.

- For checking and savings complaints, it flagged 25 unusual days out of 639
- For money transfer complaints, it flagged 5 unusual days out of 639
- Both categories showed a real, sharp spike around January 15 to 18, 2025. I looked for a specific news story that would explain it and didn't find one, but I did find that 2025 saw a big overall rise in these complaint types nationally, so this spike sits inside a real larger trend. I'm being upfront that I found something real but can't fully explain the exact cause myself. 

### What the agent said when I actually asked it questions

I ran real questions through the agent using the Claude API and got real answers back. A few examples:

**"Are there any complaint spikes I should be aware of?"**
The agent correctly found the same January 2025 spike the anomaly detector flagged, and on its own, without being told to, it noticed that several of the complaints from that exact week mentioned Zelle transfer problems. That's a lead I hadn't found myself when I researched the spike earlier. It also correctly noticed a couple of unusually low volume days and flagged them as possible data quality issues rather than assuming they meant something.

**"What are the most common issues with money transfers?"**
The agent grouped the complaints into four clear themes on its own: SIM swap fraud, missing fraud warnings, transfers sent to the wrong account with no way to reverse them, and hidden fees on currency exchange. It cited the specific complaints behind each theme.

Full text of these answers is saved in `data/processed/example_outputs.md`.

## What's real and what needs a key to run

Everything except the final answer writing step is fully built and already tested, no key needed:
- The data cleaning
- The search system
- The anomaly detector
- The evaluation numbers above

The answer writing step uses the Claude API, which needs an API key (not included, for obvious reasons). If you want to run it yourself:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

Without a key, the agent still runs, still searches, still checks for anomalies, it just skips the last step of writing a full answer.

## Project layout

```
fin-complaint-signal-agent/
├── data/
│   ├── raw/                  the original, untouched source files
│   └── processed/            cleaned data, eval results, example outputs
├── scripts/
│   └── scripts_clean_data.py cleans and filters the raw data
├── src/
│   ├── retrieval.py          builds the search index and searches it
│   ├── evaluate_retrieval.py tests how good the search is
│   ├── anomaly_detection.py  finds unusual days in the complaint volume
│   ├── agent.py               ties everything together and calls Claude
│   └── api.py                  a simple web API for the whole thing
├── Dockerfile
├── requirements.txt
└── README.md
```

## How to run it

```bash
pip install -r requirements.txt

# Step 1, clean the data (already done, output is already in data/processed)
python scripts/scripts_clean_data.py

# Step 2, build the search index (not saved in this repo since it's large, so build it locally)
cd src
python retrieval.py

# Step 3, run the anomaly detector
python anomaly_detection.py

# Step 4, run the agent (add your API key first if you want a full written answer)
python agent.py "your question here"

# Step 5, or run it as a web API
uvicorn api:app --reload
```

Or with Docker:
```bash
docker build -t complaint-agent .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=your_key complaint-agent
```

## Things I want to be upfront about

- The anomaly detector flags statistical outliers, not confirmed real world events. I'm confident the January 2025 spike is real because the numbers are so far outside normal (way more than double the usual range), not because I found a news article proving it.
- The search system only has access to complaint text written by customers, not internal company documents, so its explanations are limited to what customers themselves wrote.
- One of my evaluation checks (whether the system could find a complaint's own original text when searching using a snippet of that same text) came back at 89.5 percent instead of close to 100 percent. That's expected, since the corpus has a lot of very similar complaints, so a short snippet sometimes matches a near duplicate complaint just as well as the original one.

# Extractive Text Summarizer (TextRank)

A self-contained **NLP demo** that summarizes a document by ranking its sentences with **TextRank** — PageRank run over a TF-IDF sentence-similarity graph.

## Real-world context
Summarization turns long reports, articles or support threads into a few high-signal sentences — useful for content, research and competitive monitoring. This demo implements the classic **TextRank** algorithm from scratch (PageRank power-iteration in NumPy, no graph library).

## What it does
1. Splits the input into sentences.
2. Builds a **TF-IDF** vector per sentence (L2-normalized → dot product = cosine similarity).
3. Forms a sentence-similarity graph and runs **PageRank** (power iteration) to score sentences.
4. Selects the top ~30% (in original order) as the extractive summary, with a compression ratio.

## Results (reproducible)
See [`results/summary.md`](results/summary.md) — summary + full sentence ranking.

## Run
```bash
pip install -r requirements.txt
python summarizer_demo.py
```
Replace the `ARTICLE` string (or load a `.txt`) to summarize your own document.

## Stack
Python · NumPy · scikit-learn (TF-IDF)

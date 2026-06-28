"""Extractive text summarizer (TextRank) — TF-IDF sentence similarity + PageRank.

Self-contained (numpy + scikit-learn's TfidfVectorizer), no network. Implements the
PageRank power-iteration directly in NumPy (no networkx). Saves results/summary.md.
"""
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ARTICLE = """
Autonomous AI agents are reshaping how digital-marketing teams operate. Instead of a
single prompt-and-response interaction, an agent plans a goal, calls tools, observes the
result, and iterates until the task is done. This loop lets a system carry out
multi-step campaigns with limited human supervision.

A typical marketing agent stack combines a large language model for reasoning, a
retrieval layer for brand and product knowledge, and a set of tools such as analytics
queries, content generators, and scheduling APIs. The language model decides which tool
to call next; the retrieval layer keeps answers grounded in the company's own data.

Orchestration frameworks like LangChain, AutoGen and CrewAi coordinate several agents
that specialize in different roles. One agent might draft copy, another might review it
for tone and compliance, and a third might schedule the post and watch the metrics.
Dividing the work this way improves reliability and makes each step auditable.

The main risks are hallucination, cost, and loss of control. Grounding the model in
retrieved data reduces hallucination, while caching and smaller models cut cost. Human
review gates keep an operator in the loop for anything that is published externally.

In practice, the value of an agent is measured the same way as any marketing tool: does
it lower the time to produce a campaign, and does it move a business metric. Early
deployments report large reductions in turnaround time, with humans shifting from
execution to supervision and strategy.
"""


def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    sents = re.split(r"(?<=[.!?]) +", text)
    return [s.strip() for s in sents if len(s.split()) >= 5]


def textrank(sentences, d=0.85, iters=60):
    # TF-IDF (L2-normalized) → dot product = cosine similarity
    tfidf = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    sim = (tfidf @ tfidf.T).toarray()
    np.fill_diagonal(sim, 0.0)
    row_sum = sim.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    M = sim / row_sum                      # row-stochastic transition matrix
    n = len(sentences)
    scores = np.ones(n) / n
    for _ in range(iters):                 # PageRank power iteration
        scores = (1 - d) / n + d * (M.T @ scores)
    return scores


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(here, "results")
    os.makedirs(res, exist_ok=True)

    sents = split_sentences(ARTICLE)
    scores = textrank(sents)
    k = max(3, round(len(sents) * 0.30))
    top_idx = sorted(np.argsort(scores)[-k:])         # keep original order
    summary = " ".join(sents[i] for i in top_idx)

    words_in = sum(len(s.split()) for s in sents)
    words_out = len(summary.split())
    ratio = words_out / words_in

    with open(os.path.join(res, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Text Summarizer — Output\n\n")
        f.write(f"Source: **{len(sents)} sentences / {words_in} words** → "
                f"summary: **{k} sentences / {words_out} words** "
                f"(**{ratio:.0%}** of the original).\n\n")
        f.write("## Extractive summary (top-ranked sentences)\n")
        for i in top_idx:
            f.write(f"- {sents[i]}\n")
        f.write("\n## Sentence ranking (TextRank score)\n")
        for rank, i in enumerate(np.argsort(scores)[::-1], 1):
            f.write(f"{rank}. ({scores[i]:.4f}) {sents[i][:80]}...\n")

    print(f"sentences={len(sents)} summary={k} compression={ratio:.0%}")
    print("self-check:", "OK" if 0 < k < len(sents) and words_out < words_in else "FAIL")


if __name__ == "__main__":
    main()

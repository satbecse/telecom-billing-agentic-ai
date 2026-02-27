"""
Report generator for the RAG Evaluation Framework.

Reads raw evaluation results and generates a clean comparison table.

Usage:
    python -m app.evaluation.report <results_file.json>
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


def generate_report(results: List[Dict[str, Any]]) -> str:
    """Generate a formatted comparison report from evaluation results."""

    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append("  RAG EVALUATION COMPARISON REPORT")
    lines.append("=" * 90)

    # ── Summary Table ─────────────────────────────────────────────────
    combo_scores = defaultdict(lambda: {"faithfulness": [], "relevancy": [], "correctness": []})

    for r in results:
        key = (r.get("rag_strategy", "Unknown"), r.get("chunking", "default"))
        combo_scores[key]["faithfulness"].append(r["faithfulness"])
        combo_scores[key]["relevancy"].append(r["relevancy"])
        combo_scores[key]["correctness"].append(r["correctness"])

    lines.append("")
    lines.append("  SUMMARY (Average scores across all queries)")
    lines.append("  " + "-" * 86)
    lines.append(f"  {'RAG Strategy':<16} {'Chunking':<14} {'Faithfulness':>13} {'Relevancy':>11} {'Correctness':>13} {'Avg Score':>11}")
    lines.append("  " + "-" * 86)

    best_avg = 0
    best_combo = ""

    sorted_combos = sorted(combo_scores.items(), key=lambda x: (x[0][1], x[0][0]))

    for (rag, chunking), scores in sorted_combos:
        n = len(scores["faithfulness"])
        avg_f = sum(scores["faithfulness"]) / n
        avg_r = sum(scores["relevancy"]) / n
        avg_c = sum(scores["correctness"]) / n
        avg_total = (avg_f + avg_r + avg_c) / 3

        if avg_total > best_avg:
            best_avg = avg_total
            best_combo = f"{rag} + {chunking}"

        lines.append(f"  {rag:<16} {chunking:<14} {avg_f:>13.3f} {avg_r:>11.3f} {avg_c:>13.3f} {avg_total:>11.3f}")

    lines.append("  " + "-" * 86)
    lines.append(f"\n  WINNER: {best_combo} (avg score: {best_avg:.3f})")

    # ── Per-Query Breakdown ──────────────────────────────────────────
    lines.append("")
    lines.append("")
    lines.append("  PER-QUERY BREAKDOWN (best combo per query)")
    lines.append("  " + "-" * 86)
    lines.append(f"  {'#':<4} {'Query':<38} {'Best Combo':<26} {'F':>5} {'R':>5} {'C':>5} {'Avg':>6}")
    lines.append("  " + "-" * 86)

    query_results = defaultdict(list)
    queries_order = []
    for r in results:
        q = r["query"]
        if q not in queries_order:
            queries_order.append(q)
        query_results[q].append(r)

    for qi, query in enumerate(queries_order, 1):
        # Find the best-scoring combo for this query
        best_r = max(query_results[query], key=lambda x: x["avg_score"])
        combo_label = f"{best_r.get('rag_strategy','?')}+{best_r.get('chunking','?')}"
        avg = best_r["avg_score"]
        q_short = query[:36] + "..." if len(query) > 36 else query
        lines.append(
            f"  Q{qi:<3} {q_short:<38} {combo_label:<26} "
            f"{best_r['faithfulness']:>5.2f} {best_r['relevancy']:>5.2f} {best_r['correctness']:>5.2f} {avg:>6.2f}"
        )

    lines.append("  " + "-" * 86)
    lines.append("")
    lines.append("=" * 90)

    return "\n".join(lines)


def save_report(results: List[Dict[str, Any]], output_path: str = None) -> str:
    """Generate and save the report to a file."""
    report_text = generate_report(results)

    if output_path is None:
        output_path = "data/evaluation_results/comparison_report.txt"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m app.evaluation.report <results_file.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        results = json.load(f)

    save_report(results)

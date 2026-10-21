"""
Évaluation de la qualité des réponses du RAG sur un jeu de référence.

Le corpus du jeu est ingéré dans un store en mémoire (l'index de
VECTOR_STORE_PATH n'est ni lu ni modifié), puis chaque question est posée
à l'agent, sans historique. Chaque réponse est notée :

- retrieval  : passages attendus présents dans le contexte réellement
               donné au LLM (recall) et rang du premier (MRR)
- refus      : une question sans réponse doit être déclinée, une question
               qui en a une ne doit pas l'être
- règles     : fidélité lexicale au contexte, nombres inventés,
               recouvrement avec la réponse de référence (metrics.py)
- juge LLM   : fidélité (supported / unsupported) et justesse
               (correct / partial / incorrect), en option (judge.py)

Un run produit report.json (tout le détail) et report.md (synthèse)
dans son propre dossier ; `compare` met plusieurs runs côte à côte.
"""

import dataclasses
import json
import os
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from rag_agent.config import config
from rag_agent.evaluation.dataset import CATEGORIES, GoldenDataset, GoldenQuestion
from rag_agent.evaluation.judge import CORRECTNESS_SCORES, LLMJudge
from rag_agent.evaluation.metrics import (
    evidence_recall,
    is_hallucination_by_rules,
    is_refusal,
    lexical_support,
    reciprocal_rank,
    reference_overlap,
    unsupported_numbers,
)
from rag_agent.generation.answer import ANSWERED
from rag_agent.generation.rag_agent import RAGAgent
from rag_agent.logging_config import get_logger

logger = get_logger(__name__)

REPORT_JSON = "report.json"
REPORT_MD = "report.md"


@dataclass
class QuestionResult:
    id: str
    category: str
    question: str
    reference_answer: str
    answer: str
    status: str
    seconds: float
    sources: list[str]
    refused: bool
    evidence_recall: float
    reciprocal_rank: float
    lexical_support: float
    unsupported_numbers: list[str]
    reference_overlap: float
    hallucination_rules: bool
    faithfulness: str | None = None
    faithfulness_reason: str = ""
    correctness: str | None = None
    correctness_reason: str = ""
    hallucination_judge: bool | None = None


def ingest_corpus(agent: RAGAgent, dataset: GoldenDataset) -> int:
    """Ingère les documents du jeu ; retourne le nombre de chunks indexés"""
    for path in dataset.corpus_paths:
        agent.ingest(path, source_name=os.path.basename(path))
    return len(agent.retriever.vector_store)


def evaluate_question(agent: RAGAgent, question: GoldenQuestion, top_k: int, judge: LLMJudge | None) -> QuestionResult:
    start = time.perf_counter()
    answer = agent.ask(question.question, top_k=top_k, history=[])
    seconds = time.perf_counter() - start

    texts = {c["id"]: c["text"] for c in agent.retriever.vector_store.chunks}
    context = [texts[s.chunk_id] for s in answer.sources if s.chunk_id in texts]
    answered = answer.status == ANSWERED
    refused = is_refusal(answer.text, answered)
    # Ce que la réponse peut reprendre sans rien inventer : le contexte et la question
    grounding = context + [question.question]

    if question.answerable:
        hallucination_rules = not refused and is_hallucination_by_rules(answer.text, grounding)
    else:
        hallucination_rules = not refused

    result = QuestionResult(
        id=question.id,
        category=question.category,
        question=question.question,
        reference_answer=question.reference_answer,
        answer=answer.text.strip(),
        status=answer.status,
        seconds=seconds,
        sources=[s.source for s in answer.sources],
        refused=refused,
        evidence_recall=evidence_recall(question, context) if question.answerable else 1.0,
        reciprocal_rank=reciprocal_rank(question, context) if question.answerable else 1.0,
        lexical_support=lexical_support(answer.text, grounding),
        unsupported_numbers=sorted(unsupported_numbers(answer.text, grounding)),
        reference_overlap=reference_overlap(answer.text, question.reference_answer),
        hallucination_rules=hallucination_rules,
    )

    if judge is not None:
        if answered and not refused:
            verdict = judge.faithfulness(question.question, result.answer, context)
            result.faithfulness, result.faithfulness_reason = verdict.label, verdict.reason
        if question.answerable:
            if refused:
                result.correctness, result.correctness_reason = "incorrect", "refusal"
            else:
                verdict = judge.correctness(question.question, result.answer, question.reference_answer)
                result.correctness, result.correctness_reason = verdict.label, verdict.reason
            result.hallucination_judge = not refused and result.faithfulness == "unsupported"
        else:
            result.hallucination_judge = not refused
    return result


def evaluate(
    agent: RAGAgent,
    dataset: GoldenDataset,
    top_k: int = 5,
    judge: LLMJudge | None = None,
    progress: Callable[[int, QuestionResult], None] | None = None,
) -> list[QuestionResult]:
    results = []
    for i, question in enumerate(dataset.questions, start=1):
        result = evaluate_question(agent, question, top_k, judge)
        results.append(result)
        if progress:
            progress(i, result)
    return results


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize(results: list[QuestionResult]) -> dict:
    """
    Agrégats du run. Les taux d'hallucination portent sur toutes les
    questions : une question sans réponse à laquelle le RAG répond quand même
    compte comme une hallucination.
    """
    answerable = [r for r in results if r.category != "no_answer"]
    unanswerable = [r for r in results if r.category == "no_answer"]
    responded = [r for r in results if not r.refused]
    judged = [r for r in results if r.hallucination_judge is not None]

    summary: dict = {
        "questions": len(results),
        "retrieval": {
            "evidence_recall": _mean([r.evidence_recall for r in answerable]),
            "mrr": _mean([r.reciprocal_rank for r in answerable]),
        },
        "refusals": {
            "correct_refusal_rate": _mean([float(r.refused) for r in unanswerable]),
            "false_refusal_rate": _mean([float(r.refused) for r in answerable]),
        },
        "rules": {
            "faithfulness": _mean([r.lexical_support for r in responded if r.category != "no_answer"]),
            "answer_relevance": _mean([r.reference_overlap for r in answerable]),
            "hallucination_rate": _mean([float(r.hallucination_rules) for r in results]),
        },
        "latency_s": {
            "p50": float(np.median([r.seconds for r in results])) if results else None,
            "mean": _mean([r.seconds for r in results]),
        },
    }
    if judged:
        faithfulness = [r.faithfulness for r in results if r.faithfulness is not None]
        correctness = [r.correctness for r in answerable if r.correctness is not None]
        summary["judge"] = {
            "faithfulness": _mean([float(v == "supported") for v in faithfulness]),
            "answer_correctness": _mean([CORRECTNESS_SCORES[v] for v in correctness]),
            "hallucination_rate": _mean([float(bool(r.hallucination_judge)) for r in judged]),
            "unparsed_verdicts": sum(
                1
                for r in results
                if (r.faithfulness_reason and r.faithfulness is None)
                or (r.correctness_reason and r.correctness is None)
            ),
        }

    summary["by_category"] = {}
    for category in CATEGORIES:
        group = [r for r in results if r.category == category]
        if group:
            summary["by_category"][category] = {
                "questions": len(group),
                "evidence_recall": _mean([r.evidence_recall for r in group]) if category != "no_answer" else None,
                "refusal_rate": _mean([float(r.refused) for r in group]),
                "answer_relevance": _mean([r.reference_overlap for r in group]) if category != "no_answer" else None,
                "hallucination_rate_rules": _mean([float(r.hallucination_rules) for r in group]),
                "hallucination_rate_judge": (
                    _mean([float(bool(r.hallucination_judge)) for r in group])
                    if any(r.hallucination_judge is not None for r in group)
                    else None
                ),
            }
    return summary


def run_metadata(dataset: GoldenDataset, label: str, top_k: int, model: str, judge_model: str | None) -> dict:
    return {
        "label": label,
        "date": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": {"name": dataset.name, "version": dataset.version, "fingerprint": dataset.fingerprint},
        "top_k": top_k,
        "model": os.path.basename(model),
        "judge_model": os.path.basename(judge_model) if judge_model else None,
        "config": {k: v for k, v in dataclasses.asdict(config).items() if "path" not in k and "dir" not in k},
        "machine": f"{platform.machine()}, {os.cpu_count()} CPU, Python {platform.python_version()}",
    }


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _num(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def to_markdown(meta: dict, summary: dict, results: list[QuestionResult]) -> str:
    dataset = meta["dataset"]
    lines = [
        f"# RAG evaluation — {meta['label']}",
        "",
        "_Generated by `rag-agent eval`; do not edit by hand._",
        "",
        f"- Dataset: {dataset['name']} v{dataset['version']} (fingerprint `{dataset['fingerprint']}`), "
        f"{summary['questions']} questions",
        f"- Model: {meta['model']}; judge: {meta['judge_model'] or 'none (rule-based metrics only)'}",
        f"- top_k = {meta['top_k']}, date {meta['date']}, {meta['machine']}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Evidence recall (context) | {_pct(summary['retrieval']['evidence_recall'])} |",
        f"| MRR | {_num(summary['retrieval']['mrr'])} |",
        f"| Correct refusals (unanswerable questions) | {_pct(summary['refusals']['correct_refusal_rate'])} |",
        f"| False refusals (answerable questions) | {_pct(summary['refusals']['false_refusal_rate'])} |",
        f"| Faithfulness, rules (lexical support) | {_pct(summary['rules']['faithfulness'])} |",
        f"| Answer relevance, rules (reference overlap) | {_pct(summary['rules']['answer_relevance'])} |",
        f"| Hallucination rate, rules | {_pct(summary['rules']['hallucination_rate'])} |",
    ]
    if "judge" in summary:
        judge = summary["judge"]
        lines += [
            f"| Faithfulness, judge | {_pct(judge['faithfulness'])} |",
            f"| Answer correctness, judge | {_pct(judge['answer_correctness'])} |",
            f"| Hallucination rate, judge | {_pct(judge['hallucination_rate'])} |",
            f"| Unparsed judge verdicts | {judge['unparsed_verdicts']} |",
        ]
    lines += [
        f"| Median latency per question | {_num(summary['latency_s']['p50'], 1)} s |",
        "",
        "## By category",
        "",
        "| Category | Questions | Evidence recall | Refusals | Relevance (rules) | Hallucinations (rules) "
        "| Hallucinations (judge) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category, s in summary["by_category"].items():
        lines.append(
            f"| {category} | {s['questions']} | {_pct(s['evidence_recall'])} | {_pct(s['refusal_rate'])} "
            f"| {_pct(s['answer_relevance'])} | {_pct(s['hallucination_rate_rules'])} "
            f"| {_pct(s['hallucination_rate_judge'])} |"
        )
    lines += ["", "## Questions", "", "| Id | Category | Recall | Refused | Hallucination (rules / judge) | Answer |"]
    lines.append("|---|---|---:|:---:|:---:|---|")
    for r in results:
        judge_flag = "—" if r.hallucination_judge is None else ("yes" if r.hallucination_judge else "no")
        answer = " ".join(r.answer.split()).replace("|", "\\|")
        answer = answer if len(answer) <= 160 else answer[:157] + "..."
        lines.append(
            f"| {r.id} | {r.category} | {_pct(r.evidence_recall) if r.category != 'no_answer' else '—'} "
            f"| {'yes' if r.refused else 'no'} | {'yes' if r.hallucination_rules else 'no'} / {judge_flag} | {answer} |"
        )
    return "\n".join(lines) + "\n"


def write_report(output_dir: str, meta: dict, summary: dict, results: list[QuestionResult]) -> str:
    """Écrit le run dans `output_dir`/<date>-<label>/ ; retourne ce dossier"""
    stamp = meta["date"].replace(":", "").replace("-", "").rstrip("Z")
    run_dir = os.path.join(output_dir, f"{stamp}-{meta['label']}")
    os.makedirs(run_dir, exist_ok=True)
    report = {"meta": meta, "summary": summary, "results": [dataclasses.asdict(r) for r in results]}
    with open(os.path.join(run_dir, REPORT_JSON), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, REPORT_MD), "w", encoding="utf-8") as f:
        f.write(to_markdown(meta, summary, results))
    return run_dir


def load_report(path: str) -> dict:
    """Rapport d'un run, à partir de son dossier ou de son report.json"""
    if os.path.isdir(path):
        path = os.path.join(path, REPORT_JSON)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


COMPARED_METRICS = [
    ("Evidence recall", ("retrieval", "evidence_recall"), _pct),
    ("MRR", ("retrieval", "mrr"), _num),
    ("Correct refusals", ("refusals", "correct_refusal_rate"), _pct),
    ("False refusals", ("refusals", "false_refusal_rate"), _pct),
    ("Faithfulness (rules)", ("rules", "faithfulness"), _pct),
    ("Answer relevance (rules)", ("rules", "answer_relevance"), _pct),
    ("Hallucination rate (rules)", ("rules", "hallucination_rate"), _pct),
    ("Faithfulness (judge)", ("judge", "faithfulness"), _pct),
    ("Answer correctness (judge)", ("judge", "answer_correctness"), _pct),
    ("Hallucination rate (judge)", ("judge", "hallucination_rate"), _pct),
    ("Median latency (s)", ("latency_s", "p50"), lambda v: _num(v, 1)),
]


def compare(reports: list[dict]) -> str:
    """
    Tableau Markdown des métriques de plusieurs runs côte à côte. Refuse de
    comparer des runs faits sur des jeux différents (empreintes distinctes).
    """
    fingerprints = {r["meta"]["dataset"]["fingerprint"] for r in reports}
    if len(fingerprints) > 1:
        raise ValueError(f"runs were made on different datasets ({', '.join(sorted(fingerprints))}): not comparable")

    header = "| Metric | " + " | ".join(r["meta"]["label"] for r in reports) + " |"
    lines = [header, "|---|" + "---:|" * len(reports)]
    for name, (section, key), fmt in COMPARED_METRICS:
        values = [r["summary"].get(section, {}).get(key) for r in reports]
        if all(v is None for v in values):
            continue
        lines.append(f"| {name} | " + " | ".join(fmt(v) for v in values) + " |")
    return "\n".join(lines) + "\n"

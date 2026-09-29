"""Metrics for evaluators and translators, with 95% confidence intervals."""

import math
from collections import defaultdict

import sacrebleu

from safetranslate.data.schema import ErrorItem, EvaluatorExample


def wilson(successes: int, total: int, z: float = 1.96) -> dict:
    """A proportion with its Wilson 95% confidence interval (reliable for small n)."""
    if total == 0:
        return {"rate": None, "low": None, "high": None, "n": 0}
    p = successes / total
    centre = (p + z**2 / (2 * total)) / (1 + z**2 / total)
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / (1 + z**2 / total)
    return {"rate": p, "low": centre - margin, "high": centre + margin, "n": total}


def evaluator_metrics(
    examples: list[EvaluatorExample], predictions: dict[str, list[ErrorItem]]
) -> dict[str, dict]:
    """Scores an evaluator against examples whose errors are known, per language and category.

    Recall: share of examples with errors that were flagged. False-positive rate: share of
    error-free examples that were flagged. Category accuracy: among flagged examples with
    errors, share whose predicted categories include every true category.

    Returns:
        Metrics keyed by group, e.g. "all", "ms", "ms/wrong_quantity".
    """
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for example in examples:
        predicted = predictions[example.id]
        true_categories = {e.category for e in example.errors}
        groups = ["all", example.target_lang]
        groups += [f"{example.target_lang}/{c}" for c in true_categories]
        for group in groups:
            c = counts[group]
            if true_categories:
                c["positives"] += 1
                c["caught"] += bool(predicted)
                c["category_right"] += true_categories <= {e.category for e in predicted}
            else:
                c["negatives"] += 1
                c["false_alarms"] += bool(predicted)
    results = {}
    for group, c in counts.items():
        flagged = c["caught"] + c["false_alarms"]
        precision = c["caught"] / flagged if flagged else None
        recall = c["caught"] / c["positives"] if c["positives"] else None
        results[group] = {
            "recall": wilson(c["caught"], c["positives"]),
            "false_positive_rate": wilson(c["false_alarms"], c["negatives"]),
            "precision": precision,
            "f1": 2 * precision * recall / (precision + recall) if precision and recall else None,
            "category_accuracy": wilson(c["category_right"], c["caught"]),
        }
    return results


def binary_metrics(labels: list[bool], flagged: list[bool]) -> dict:
    """Recall, false-positive rate and precision for yes/no critical-error labels."""
    caught = sum(l and f for l, f in zip(labels, flagged))
    false_alarms = sum(f and not l for l, f in zip(labels, flagged))
    return {
        "recall": wilson(caught, sum(labels)),
        "false_positive_rate": wilson(false_alarms, len(labels) - sum(labels)),
        "precision": caught / (caught + false_alarms) if caught + false_alarms else None,
    }


def rogan_gladen(apparent_rate: float, sensitivity: float, specificity: float) -> float:
    """Estimates a true rate from an imperfect detector's rate and its known accuracy.

    From epidemiology: true = (apparent + specificity - 1) / (sensitivity + specificity - 1),
    clipped to [0, 1].
    """
    denominator = sensitivity + specificity - 1
    if denominator <= 0:
        raise ValueError("detector is no better than chance, so the rate can't be corrected")
    return min(1.0, max(0.0, (apparent_rate + specificity - 1) / denominator))


def translation_scores(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    """Corpus-level spBLEU and chrF of translations against professional references."""
    # flores200 tokenisation makes BLEU work for Chinese and Tamil, and matches FLORES papers
    spbleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="flores200")
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    return {"spbleu": spbleu.score, "chrf": chrf.score}

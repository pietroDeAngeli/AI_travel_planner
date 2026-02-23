"""
NLU Evaluation Test Suite

Loads test cases from nlu_test_utterances.json and evaluates the NLU component.

Metrics:
- Intent Accuracy
- Slot Precision, Recall, F1
- Exact Match Accuracy (intent + all slots correct)
- Multi-intent Accuracy (evaluated separately, order-sensitive)

Run with:  python test/nlu_test.py [--verbose]
"""

import json
import sys
import os
import argparse
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm import make_llm
from nlu import nlu_parse
from dm import DialogueState, state_context
from intent_splitter import split_intents
from schema import INTENTS

# ---------------------------------------------------------------------------
# Path to the utterances file
# ---------------------------------------------------------------------------

UTTERANCES_FILE = os.path.join(os.path.dirname(__file__), "nlu_test_utterances.json")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(value: Any) -> Optional[str]:
    """Normalize a slot value to a comparable lowercase string, or None."""
    if value is None:
        return None
    return str(value).strip().lower()


def _call_nlu(pipe, utterance: str) -> Dict[str, Any]:
    """Run the NLU on a single utterance with a default dialogue state."""
    state = DialogueState()
    system_prompt = state_context(state)
    return nlu_parse(pipe, utterance, system_prompt, dialogue_history=None)


def _non_null_slots(slots: Dict[str, Any]) -> Dict[str, str]:
    """Return only the slots that have a non-null value, normalized."""
    return {k: _normalize(v) for k, v in slots.items() if v is not None}

# ---------------------------------------------------------------------------
# Per-case comparison
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    """Result of evaluating a single (intent, slots) prediction."""
    case_id: str
    information_type: str
    intent_correct: bool
    expected_intent: str
    predicted_intent: str
    slot_tp: int = 0          # exact (name, value) matches
    slot_fp: int = 0          # predicted but wrong or extra
    slot_fn: int = 0          # in ground truth but missing / wrong
    exact_match: bool = False # intent correct AND all slots correct


def _compare_single(
    expected_intent: str,
    expected_slots: Dict[str, Any],
    predicted_intent: str,
    predicted_slots: Dict[str, Any],
) -> Tuple[bool, int, int, int, bool]:
    """
    Compare a single (intent, slots) prediction against ground truth.

    Slot comparison rules:
    - Slot name must match exactly.
    - Slot value must match exactly (case-insensitive after normalization).
    - No partial credit for incorrect values.

    Returns (intent_ok, tp, fp, fn, exact_match).
    """
    intent_ok = (predicted_intent == expected_intent)

    gt = _non_null_slots(expected_slots)
    pred = _non_null_slots(predicted_slots)

    tp = fp = fn = 0

    # True positives & false negatives
    for slot_name, gt_val in gt.items():
        pred_val = pred.get(slot_name)
        if pred_val is not None and pred_val == gt_val:
            tp += 1
        else:
            fn += 1

    # False positives: predicted slots not in ground truth or with wrong value
    for slot_name, pred_val in pred.items():
        if slot_name not in gt:
            fp += 1
        elif _normalize(gt[slot_name]) != pred_val:
            # Value mismatch — already counted as fn; also counts as fp
            fp += 1

    exact = intent_ok and (fp == 0) and (fn == 0)
    return intent_ok, tp, fp, fn, exact

# ---------------------------------------------------------------------------
# Metrics aggregator
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    """Accumulates evaluation metrics across test cases."""
    single_results: List[CaseResult] = field(default_factory=list)
    multi_results: List[Dict[str, Any]] = field(default_factory=list)

    def add_single(self, result: CaseResult):
        self.single_results.append(result)

    def add_multi(self, case_id: str, info_type: str,
                  per_intent_results: List[CaseResult], order_correct: bool):
        self.multi_results.append({
            "case_id": case_id,
            "information_type": info_type,
            "per_intent": per_intent_results,
            "order_correct": order_correct,
        })

    # --- aggregation helpers ---

    @staticmethod
    def _agg(results: List[CaseResult]):
        total = len(results)
        intent_correct = sum(r.intent_correct for r in results)
        exact_match = sum(r.exact_match for r in results)
        tp = sum(r.slot_tp for r in results)
        fp = sum(r.slot_fp for r in results)
        fn = sum(r.slot_fn for r in results)
        return total, intent_correct, exact_match, tp, fp, fn

    @staticmethod
    def _prf(tp: int, fp: int, fn: int):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    # --- printing ---

    def print_summary(self):
        self._print_single_summary()
        self._print_per_type_summary()
        self._print_per_intent_summary()
        self._print_multi_summary()
        self._print_overall_summary()

    def _print_single_summary(self):
        total, ic, em, tp, fp, fn = self._agg(self.single_results)
        p, r, f1 = self._prf(tp, fp, fn)

        print("\n" + "=" * 80)
        print("SINGLE-INTENT EVALUATION")
        print("=" * 80)
        print(f"  Total test cases      : {total}")
        if total:
            print(f"  Intent Accuracy       : {ic}/{total}  ({ic/total:.2%})")
            print(f"  Exact Match Accuracy  : {em}/{total}  ({em/total:.2%})")
        print(f"  Slot Precision        : {p:.4f}")
        print(f"  Slot Recall           : {r:.4f}")
        print(f"  Slot F1               : {f1:.4f}")
        print(f"  (TP={tp}  FP={fp}  FN={fn})")

    def _print_per_type_summary(self):
        print("\n" + "-" * 80)
        print("BREAKDOWN BY information_type (single-intent)")
        print("-" * 80)
        by_type: Dict[str, List[CaseResult]] = defaultdict(list)
        for r in self.single_results:
            by_type[r.information_type].append(r)

        header = f"{'Type':<22} {'#':>4} {'IntAcc':>8} {'ExactM':>8} {'SlotP':>8} {'SlotR':>8} {'SlotF1':>8}"
        print(header)
        print("-" * len(header))
        for info_type in sorted(by_type):
            results = by_type[info_type]
            total, ic, em, tp, fp, fn = self._agg(results)
            p, r, f1 = self._prf(tp, fp, fn)
            int_acc = ic / total if total else 0
            ex_acc = em / total if total else 0
            print(f"{info_type:<22} {total:>4} {int_acc:>8.2%} {ex_acc:>8.2%} {p:>8.4f} {r:>8.4f} {f1:>8.4f}")

    def _print_per_intent_summary(self):
        print("\n" + "-" * 80)
        print("BREAKDOWN BY INTENT (single-intent)")
        print("-" * 80)

        by_expected: Dict[str, List[CaseResult]] = defaultdict(list)
        for r in self.single_results:
            by_expected[r.expected_intent].append(r)

        # Intent-classification P/R/F1
        intent_tp: Dict[str, int] = defaultdict(int)
        intent_fp: Dict[str, int] = defaultdict(int)
        intent_fn: Dict[str, int] = defaultdict(int)

        for r in self.single_results:
            if r.predicted_intent == r.expected_intent:
                intent_tp[r.expected_intent] += 1
            else:
                intent_fn[r.expected_intent] += 1
                intent_fp[r.predicted_intent] += 1

        header = f"{'Intent':<25} {'Supp':>5} {'IntP':>8} {'IntR':>8} {'IntF1':>8} {'SlotF1':>8} {'ExactM':>8}"
        print(header)
        print("-" * len(header))

        for intent in sorted(set(r.expected_intent for r in self.single_results)):
            supp = len(by_expected[intent])
            tp_i = intent_tp[intent]
            fp_i = intent_fp.get(intent, 0)
            fn_i = intent_fn[intent]
            p_i = tp_i / (tp_i + fp_i) if (tp_i + fp_i) else 0
            r_i = tp_i / (tp_i + fn_i) if (tp_i + fn_i) else 0
            f1_i = 2 * p_i * r_i / (p_i + r_i) if (p_i + r_i) else 0

            _, _, _, stp, sfp, sfn = self._agg(by_expected[intent])
            _, _, sf1 = self._prf(stp, sfp, sfn)

            em_count = sum(r.exact_match for r in by_expected[intent])
            em_rate = em_count / supp if supp else 0

            print(f"{intent:<25} {supp:>5} {p_i:>8.2%} {r_i:>8.2%} {f1_i:>8.4f} {sf1:>8.4f} {em_rate:>8.2%}")

    def _print_multi_summary(self):
        if not self.multi_results:
            return

        print("\n" + "=" * 80)
        print("MULTI-INTENT EVALUATION")
        print("=" * 80)

        total = len(self.multi_results)
        order_correct = sum(m["order_correct"] for m in self.multi_results)

        all_per_intent: List[CaseResult] = []
        for m in self.multi_results:
            all_per_intent.extend(m["per_intent"])

        n_intents = len(all_per_intent)
        _, ic, em, tp, fp, fn = self._agg(all_per_intent)
        p, r, f1 = self._prf(tp, fp, fn)

        fully_correct = sum(
            1 for m in self.multi_results
            if m["order_correct"] and all(cr.exact_match for cr in m["per_intent"])
        )

        print(f"  Total multi-intent cases        : {total}")
        print(f"  Total sub-intents evaluated      : {n_intents}")
        if n_intents:
            print(f"  Intent Accuracy (per sub-intent) : {ic}/{n_intents}  ({ic/n_intents:.2%})")
        if total:
            print(f"  Order-correct cases              : {order_correct}/{total}  ({order_correct/total:.2%})")
            print(f"  Fully correct cases              : {fully_correct}/{total}  ({fully_correct/total:.2%})")
        print(f"  Slot Precision (sub-intents)     : {p:.4f}")
        print(f"  Slot Recall    (sub-intents)     : {r:.4f}")
        print(f"  Slot F1        (sub-intents)     : {f1:.4f}")
        print(f"  (TP={tp}  FP={fp}  FN={fn})")

    def _print_overall_summary(self):
        all_results = list(self.single_results)
        for m in self.multi_results:
            all_results.extend(m["per_intent"])

        total, ic, em, tp, fp, fn = self._agg(all_results)
        p, r, f1 = self._prf(tp, fp, fn)

        print("\n" + "=" * 80)
        print("OVERALL SUMMARY (single + multi sub-intents)")
        print("=" * 80)
        print(f"  Total evaluated        : {total}")
        if total:
            print(f"  Intent Accuracy        : {ic}/{total}  ({ic/total:.2%})")
            print(f"  Exact Match Accuracy   : {em}/{total}  ({em/total:.2%})")
        print(f"  Slot Precision         : {p:.4f}")
        print(f"  Slot Recall            : {r:.4f}")
        print(f"  Slot F1                : {f1:.4f}")

    # --- export to JSON ---

    def to_dict(self) -> Dict[str, Any]:
        all_results = list(self.single_results)
        for m in self.multi_results:
            all_results.extend(m["per_intent"])
        total, ic, em, tp, fp, fn = self._agg(all_results)
        p, r, f1 = self._prf(tp, fp, fn)

        return {
            "overall": {
                "total": total,
                "intent_accuracy": ic / total if total else 0,
                "exact_match_accuracy": em / total if total else 0,
                "slot_precision": p,
                "slot_recall": r,
                "slot_f1": f1,
            },
            "single_intent": [
                {
                    "id": cr.case_id,
                    "type": cr.information_type,
                    "intent_correct": cr.intent_correct,
                    "exact_match": cr.exact_match,
                    "expected_intent": cr.expected_intent,
                    "predicted_intent": cr.predicted_intent,
                }
                for cr in self.single_results
            ],
            "multi_intent": [
                {
                    "id": m["case_id"],
                    "order_correct": m["order_correct"],
                    "per_intent": [
                        {
                            "intent_correct": cr.intent_correct,
                            "exact_match": cr.exact_match,
                            "expected_intent": cr.expected_intent,
                            "predicted_intent": cr.predicted_intent,
                        }
                        for cr in m["per_intent"]
                    ],
                }
                for m in self.multi_results
            ],
        }

# ---------------------------------------------------------------------------
# Evaluation runners
# ---------------------------------------------------------------------------

def evaluate_single(pipe, case: Dict[str, Any], verbose: bool = False) -> CaseResult:
    """Evaluate a single-intent test case."""
    case_id = case["id"]
    info_type = case["information_type"]
    utterance = case["utterance"]
    gt = case["ground_truth"]
    expected_intent = gt["intent"]
    expected_slots = gt.get("slots", {})

    nlu_out = _call_nlu(pipe, utterance)
    predicted_intent = nlu_out.get("intent", "OOD")
    predicted_slots = nlu_out.get("slots", {})

    intent_ok, tp, fp, fn, exact = _compare_single(
        expected_intent, expected_slots,
        predicted_intent, predicted_slots,
    )

    result = CaseResult(
        case_id=case_id,
        information_type=info_type,
        intent_correct=intent_ok,
        expected_intent=expected_intent,
        predicted_intent=predicted_intent,
        slot_tp=tp, slot_fp=fp, slot_fn=fn,
        exact_match=exact,
    )

    if verbose:
        mark = "✓" if exact else ("~" if intent_ok else "✗")
        print(f"  [{mark}] {case_id:<35} expected={expected_intent:<22} got={predicted_intent:<22} EM={exact}")
        if not exact:
            gt_disp = _non_null_slots(expected_slots)
            pred_disp = _non_null_slots(predicted_slots)
            print(f"       gt_slots  = {gt_disp}")
            print(f"       pred_slots= {pred_disp}")

    return result


def evaluate_multi(pipe, case: Dict[str, Any], verbose: bool = False) -> Tuple[List[CaseResult], bool]:
    """
    Evaluate a multi-intent test case.

    Uses the intent_splitter to split the utterance, then runs NLU on each part.
    Compares predictions to ground truth in order (order must match).
    """
    case_id = case["id"]
    utterance = case["utterance"]
    gt_list: List[Dict[str, Any]] = case["ground_truth"]
    expected_count = len(gt_list)

    # Split using the LLM-based intent splitter
    first_sentence, pending = split_intents(pipe, utterance)
    all_sentences = [first_sentence] + pending

    # Run NLU on each split sentence
    predictions = []
    for sentence in all_sentences:
        nlu_out = _call_nlu(pipe, sentence)
        predictions.append(nlu_out)

    # Compare in order: pad with empty predictions if splitter returned fewer
    per_intent_results: List[CaseResult] = []
    order_correct = True

    for i, gt in enumerate(gt_list):
        expected_intent = gt["intent"]
        expected_slots = gt.get("slots", {})

        if i < len(predictions):
            pred = predictions[i]
            predicted_intent = pred.get("intent", "OOD")
            predicted_slots = pred.get("slots", {})
        else:
            predicted_intent = "__MISSING__"
            predicted_slots = {}

        intent_ok, tp, fp, fn, exact = _compare_single(
            expected_intent, expected_slots,
            predicted_intent, predicted_slots,
        )

        if not intent_ok:
            order_correct = False

        cr = CaseResult(
            case_id=f"{case_id}[{i}]",
            information_type="multi_intent",
            intent_correct=intent_ok,
            expected_intent=expected_intent,
            predicted_intent=predicted_intent,
            slot_tp=tp, slot_fp=fp, slot_fn=fn,
            exact_match=exact,
        )
        per_intent_results.append(cr)

    # If splitter produced more or fewer sentences than expected, order is wrong
    if len(predictions) != expected_count:
        order_correct = False

    if verbose:
        all_exact = all(cr.exact_match for cr in per_intent_results)
        mark = "✓" if (order_correct and all_exact) else "✗"
        print(f"  [{mark}] {case_id:<35} split={len(all_sentences)}  expected={expected_count}  order_ok={order_correct}")
        for i, cr in enumerate(per_intent_results):
            sub_mark = "✓" if cr.exact_match else "✗"
            print(f"       [{sub_mark}] sub-intent {i}: expected={cr.expected_intent:<22} got={cr.predicted_intent:<22} EM={cr.exact_match}")

    return per_intent_results, order_correct

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation(verbose: bool = False):
    # Load test cases
    print(f"Loading test cases from {UTTERANCES_FILE} ...")
    with open(UTTERANCES_FILE, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    print(f"Loaded {len(test_cases)} test cases.\n")

    # Separate single vs multi
    single_cases = [tc for tc in test_cases if not isinstance(tc["ground_truth"], list)]
    multi_cases = [tc for tc in test_cases if isinstance(tc["ground_truth"], list)]
    print(f"  Single-intent : {len(single_cases)}")
    print(f"  Multi-intent  : {len(multi_cases)}\n")

    # Load LLM
    print("Loading LLM pipeline ...")
    pipe = make_llm()
    if pipe is None:
        print("ERROR: Could not load LLM pipeline. Aborting.")
        sys.exit(1)
    print("LLM loaded.\n")

    metrics = Metrics()

    # --- Single-intent evaluation ---
    print("=" * 80)
    print("EVALUATING SINGLE-INTENT CASES")
    print("=" * 80)
    for case in single_cases:
        result = evaluate_single(pipe, case, verbose=verbose)
        metrics.add_single(result)

    # --- Multi-intent evaluation ---
    if multi_cases:
        print("\n" + "=" * 80)
        print("EVALUATING MULTI-INTENT CASES")
        print("=" * 80)
        for case in multi_cases:
            per_intent, order_ok = evaluate_multi(pipe, case, verbose=verbose)
            metrics.add_multi(case["id"], "multi_intent", per_intent, order_ok)

    # --- Print summary ---
    metrics.print_summary()

    # --- Save results ---
    results_path = os.path.join(os.path.dirname(__file__), "nlu_test_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLU Evaluation")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-case details")
    args = parser.parse_args()
    run_evaluation(verbose=args.verbose)

"""Evaluation functions."""

import random
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from .normalization import normalize_token_with_map
from .preprocessing import get_tags, get_token_alternatives


def extract_test_samples(test_set):
    """
    Input:
        test_set: list of tagged test sequences
    Output:
        filtered_sequences: longest sequences of at least four complete
            tokens, with one interpretation selected per token
    """
    filtered_sequences = []

    for sequence in test_set:
        longest_sequence = []
        current_sequence = []

        for token, tag in sequence:
            alternatives = get_token_alternatives(token)
            complete_alternatives = [
                alternative
                for alternative, alternative_tag in get_tags(alternatives)
                if alternative_tag == "<com>"
            ]

            if complete_alternatives:
                selected_token = complete_alternatives[0]
                current_sequence.append(selected_token)

                if len(current_sequence) > len(longest_sequence):
                    longest_sequence = current_sequence[:]
            else:
                current_sequence = []

        if len(longest_sequence) >= 4:
            filtered_sequences.append(longest_sequence)

    return filtered_sequences


def alter_token(token, seed):
    """
    Input:
        token: token to modify
        seed: random seed (0–4)
    Output:
        token with one or more missing-character symbols
    """
    random.seed(seed)
    token = list(token)
    max_changes = len(token) - 1
    number_of_changes = 0
    replacement = random.choice(["-", "…"])

    if replacement == "-":
        while number_of_changes != max_changes:
            pos = random.randint(0, len(token) - 1)
            token[pos] = "-"
            number_of_changes += 1
            add_changes = random.choice(["yes", "no"])
            if add_changes == "no":
                break
        return "".join(token)

    if replacement == "…":
        start = random.randint(0, len(token) - 2)
        end = random.randint(start + 1, len(token) - 1)
        return "".join(token[:start]) + replacement + "".join(token[end:])


def mask_tokens(sequence, seed, mask_fraction=1 / 6):
    """
    Input:
        sequence: list of tokens
        seed: random seed (0–4)
        mask_fraction: fraction of tokens to mask entirely as <mis>
    Output:
        masked_sequence: sequence with a subset turned into <mis> (single '…')
    """
    random.seed(seed)
    seq = sequence[:]
    n = len(seq)
    num_to_mask = max(1, int(round(n * mask_fraction)))

    indices = list(range(n))
    to_mask = set(random.sample(indices, min(num_to_mask, n)))

    for idx in to_mask:
        seq[idx] = "…"
    return seq


def alter_sequence(sequence, seed, masked_indices=None):
    """
    Input:
        sequence: list of tokens
        seed: random seed (0–4)
        masked_indices: optional set of indices that are already <mis> and must be skipped
    Output:
        altered sequence with incomplete tokens
    """
    random.seed(seed)
    viable_tokens = []
    sequence = sequence[:]
    for index, token in enumerate(sequence):
        if masked_indices and index in masked_indices:
            continue
        if len(token) > 1:
            viable_tokens.append(index)

    if len(sequence) in [4]:
        num_tokens_to_alter = 1
    elif len(sequence) in [5, 6]:
        num_tokens_to_alter = 2
    else:
        num_tokens_to_alter = 3

    tokens_to_alter = random.sample(
        viable_tokens, min(num_tokens_to_alter, len(viable_tokens))
    )

    for i in tokens_to_alter:
        sequence[i] = alter_token(sequence[i], seed)
    return sequence


def alter_sequence_with_optional_mask(
    sequence, seed, mask_missing=True, mask_fraction=1 / 6
):
    """
    Input:
        sequence: list of tokens
        seed: random seed (0–4)
        mask_missing: whether to first create <mis> tokens
        mask_fraction: fraction of tokens to mask if mask_missing=True
    Output:
        sequence after optional masking and subsequent <inc> noise
    """
    if not mask_missing:
        return alter_sequence(sequence, seed)

    masked = mask_tokens(sequence, seed, mask_fraction=mask_fraction)
    masked_indices = {i for i, t in enumerate(masked) if t in {"-", "…"}}
    return alter_sequence(masked, seed, masked_indices=masked_indices)


def alter_sequences(sequences, mask_missing=True, mask_fraction=1 / 6):
    """
    Input:
        sequences: list of test sequences
        mask_missing: whether to first create <mis> tokens
        mask_fraction: fraction to mask as <mis> when mask_missing=True
    Output:
        altered_sequences: list of synthetically modified test sequences
    """
    altered_sequences = []
    for sequence in sequences:
        for seed in range(5):
            altered_sequence = alter_sequence_with_optional_mask(
                sequence, seed, mask_missing=mask_missing, mask_fraction=mask_fraction
            )
            altered_sequences.append(altered_sequence)
    return altered_sequences


def calculate_metrics(gold_pred_list, original_tagged_list, eval_scope="both"):
    """
    Input:
        gold_pred_list: list of tuples (gold sequence, predicted sequence)
        original_tagged_list: list of original tagged sequences (with <com>/<inc>/<mis>)
        eval_scope: 'inc', 'mis', or 'both'
    Output:
        prediction_coverage, accuracy, MRR, and raw counts filtered by eval_scope
    """
    correct_predictions = 0
    incorrect_predictions = 0
    non_predictions = 0
    reciprocal_ranks = []

    for idx, item in enumerate(gold_pred_list):
        gold, pred = item
        orig_tags = [tag for _, tag in original_tagged_list[idx]]

        for i, predictions in enumerate(pred):
            if predictions[1] == "<mod>":
                original_tag = orig_tags[i]

                if eval_scope == "inc" and original_tag != "<inc>":
                    continue
                if eval_scope == "mis" and original_tag != "<mis>":
                    continue

                if len(predictions[0]) == 0:
                    non_predictions += 1
                else:
                    if gold[i] in predictions[0]:
                        correct_predictions += 1
                        rank = predictions[0].index(gold[i]) + 1
                        reciprocal_ranks.append(1.0 / rank)
                    else:
                        incorrect_predictions += 1

    predicted = correct_predictions + incorrect_predictions
    to_predict = correct_predictions + incorrect_predictions + non_predictions

    prediction_coverage = (
        (to_predict - non_predictions) / to_predict if to_predict else 0
    )
    accuracy = correct_predictions / predicted if predicted else 0
    MRR = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0

    return (
        prediction_coverage,
        accuracy,
        MRR,
        correct_predictions,
        incorrect_predictions,
        non_predictions,
    )


def _normalize_sequence(
    seq: List[str], runic_to_normalized: Dict[str, str]
) -> List[str]:
    """
    Input:
        seq: list of tokens
        runic_to_normalized: mapping
    Output:
        normalized_seq: list of normalized tokens (no change if token not in mapping)
    """
    return [normalize_token_with_map(t, runic_to_normalized) for t in seq]


def calculate_mis_metrics_with_normalization(
    gold_pred_list: List[Tuple[List[str], List[Tuple[Union[str, List[str]], str]]]],
    original_tagged_list: List[List[Tuple[str, str]]],
    runic_to_normalized: Dict[str, str],
    use_normalization: bool = True,
) -> Dict[str, float]:
    """
    Input:
        gold_pred_list: gold and predicted token sequences
        original_tagged_list: original tagged sequences
        runic_to_normalized: mapping from runic to normalized forms
        use_normalization: whether to compare normalized token forms
    Output:
        metrics: coverage, accuracy, and conditional MRR for the selected mode
    """
    correct = 0
    incorrect = 0
    nonpred = 0
    reciprocal_ranks: List[float] = []

    for idx, (gold, pred) in enumerate(gold_pred_list):
        original_tags = [tag for _, tag in original_tagged_list[idx]]

        for token_idx, (prediction, prediction_tag) in enumerate(pred):
            if prediction_tag != "<mod>" or original_tags[token_idx] != "<mis>":
                continue

            candidates = prediction if isinstance(prediction, list) else []
            if not candidates:
                nonpred += 1
                continue

            target = gold[token_idx]
            if use_normalization:
                target = _normalize_sequence([target], runic_to_normalized)[0]
                candidates_to_score = [
                    _normalize_sequence([candidate], runic_to_normalized)[0]
                    for candidate in candidates
                ]
            else:
                candidates_to_score = candidates

            if target in candidates_to_score:
                correct += 1
                rank = candidates_to_score.index(target) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                incorrect += 1

    predicted = correct + incorrect
    total = predicted + nonpred
    coverage = predicted / total if total else 0.0
    accuracy = correct / predicted if predicted else 0.0
    conditional_mrr = (
        sum(reciprocal_ranks) / len(reciprocal_ranks)
        if reciprocal_ranks else 0.0
    )

    suffix = "norm" if use_normalization else "literal"
    return {
        f"prediction_coverage_{suffix}": coverage,
        f"accuracy_{suffix}": accuracy,
        f"MRR_{suffix}": conditional_mrr,
    }

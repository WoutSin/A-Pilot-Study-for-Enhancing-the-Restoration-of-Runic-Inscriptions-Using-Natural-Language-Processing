import copy
import json
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from typing import Dict, Iterable, List, Optional, Tuple, Union


# Core utilities

def read_excel_to_dataframe(file_path):
    """
    Input:
        file_path: path to an Excel file containing transliterated runic inscriptions
    Output:
        df: dataframe containing transliterated runic texts
    """
    df = pd.read_excel(file_path)
    return df


def remove_punctuation(input_string, punctuation):
    """
    Input:
        input_string: runic sequence (string)
        punctuation: string of non-alphabetical characters to be removed
    Output:
        no_punct: runic sequence without unnecessary punctuation (string)
    """
    translator = str.maketrans("", "", punctuation)
    no_punct = input_string.translate(translator)
    return no_punct


def print_non_alpha(runic_list):
    """
    Input:
        runic_list: list of runic inscriptions
    Output:
        non_alpha_chars: string of non-alphabetical characters to be removed
    """
    non_alpha_chars = set()
    for item in runic_list:
        non_alpha_chars.update(
            [
                char
                for char in item
                if not char.isalnum() and char not in ["-", "…", " "]
            ]
        )
    return "".join(non_alpha_chars)


def tokenize_text(input_string):
    """
    Input:
        input_string: runic sequence (string)
    Output:
        tokens: list of tokens in the runic sequence
    """
    return [token for token in input_string.split(" ") if token]


def get_tags(runic_list):
    """
    Input:
        runic_list: tokenized runic sequence
    Output:
        output_list: list of tuples (token, tag)
    Tags:
        <com>: complete tokens (no missing characters)
        <inc>: incomplete tokens (some characters missing)
        <mis>: missing tokens (all characters missing)
    """
    special_chars = ["-", "…"]
    output_list = []

    for item in runic_list:
        if any(char in item for char in special_chars) and any(
            char not in special_chars for char in item
        ):
            output_list.append((item, "<inc>"))
        elif all(char in special_chars for char in item):
            output_list.append((item, "<mis>"))
        else:
            output_list.append((item, "<com>"))
    return output_list


def split_data(tagged_sequences):
    """
    Input:
        tagged_sequences: list of tagged runic sequences
    Output:
        train: training subset
        test: testing subset
    """
    train, test = train_test_split(tagged_sequences, test_size=0.05, random_state=27)
    return train, test


def extract_ngram_probabilities(tagged_sequences):
    """
    Input:
        tagged_sequences: list of tagged runic sequences
    Output:
        unigrams, bigrams, trigrams: dictionaries containing n-gram probabilities
    """
    unigrams = defaultdict(int)
    bigrams = defaultdict(Counter)
    trigrams = defaultdict(Counter)

    for tagged_sequence in tagged_sequences:
        com_items = []

        for i in range(len(tagged_sequence)):
            if tagged_sequence[i][1] == "<com>":
                com_items.append(tagged_sequence[i][0])
                unigrams[tagged_sequence[i][0]] += 1
            else:
                com_items.append(None)

        for i in range(len(com_items)):
            if (
                i < len(com_items) - 1
                and com_items[i] is not None
                and com_items[i + 1] is not None
            ):
                bigrams[com_items[i]][com_items[i + 1]] += 1
            if (
                i < len(com_items) - 2
                and com_items[i] is not None
                and com_items[i + 1] is not None
                and com_items[i + 2] is not None
            ):
                trigrams[(com_items[i], com_items[i + 1])][com_items[i + 2]] += 1

    total_unigrams = sum(unigrams.values()) or 1
    for word in unigrams:
        unigrams[word] /= total_unigrams

    for word in bigrams:
        total_count = sum(bigrams[word].values()) or 1
        for next_word in bigrams[word]:
            bigrams[word][next_word] /= total_count

    for words in trigrams:
        total_count = sum(trigrams[words].values()) or 1
        for next_word in trigrams[words]:
            trigrams[words][next_word] /= total_count

    return unigrams, bigrams, trigrams


def save_probabilities_to_file(unigrams, bigrams, trigrams, filename):
    """
    Input:
        unigrams, bigrams, trigrams: n-gram probability dictionaries
        filename: output file name
    Output:
        text file with n-gram probabilities
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Unigram Probabilities:\n")
        for word, probability in unigrams.items():
            f.write(f"{word}: {probability}\n")

        f.write("\nBigram Probabilities:\n")
        for word, next_words in bigrams.items():
            for next_word, probability in next_words.items():
                f.write(f"{word} {next_word}: {probability}\n")

        f.write("\nTrigram Probabilities:\n")
        for words, next_words in trigrams.items():
            for next_word, probability in next_words.items():
                f.write(f"{words[0]} {words[1]} {next_word}: {probability}\n")


def extract_unigram_tokens(unigrams_dictionary):
    """
    Input:
        unigrams_dictionary: dictionary of unigrams and their probabilities
    Output:
        unigram_words: list of (unigram, probability) tuples
    """
    unigram_words = [
        (word, probability) for word, probability in unigrams_dictionary.items()
    ]
    return list(set(unigram_words))


def extract_bigram_tokens(bigrams_dictionary):
    """
    Input:
        bigrams_dictionary: dictionary of bigrams and probabilities
    Output:
        bigram_words: list of ((token1, token2), probability) tuples
    """
    bigram_words = []
    for word, next_words in bigrams_dictionary.items():
        for next_word, probability in next_words.items():
            bigram_words.append(((word, next_word), probability))
    return list(set(bigram_words))


def extract_trigram_tokens(trigrams_dictionary):
    """
    Input:
        trigrams_dictionary: dictionary of trigrams and probabilities
    Output:
        trigram_words: list of ((token1, token2, token3), probability) tuples
    """
    trigram_words = []
    for words, next_words in trigrams_dictionary.items():
        for next_word, probability in next_words.items():
            trigram_words.append(((words[0], words[1], next_word), probability))
    return list(set(trigram_words))


def extract_potential_tokens(tagged_sequences, unigrams, bigrams, trigrams):
    """
    Input:
        tagged_sequences: list of tagged sequences
        unigrams, bigrams, trigrams: token probability data
    Output:
        potential token dictionaries for each n-gram level
    """
    potential_tokens_from_unigrams = defaultdict(list)
    potential_tokens_from_bigrams = defaultdict(list)
    potential_tokens_from_trigrams = defaultdict(list)

    total_length = len(unigrams) + len(bigrams) + len(trigrams)
    pbar = tqdm(total=total_length, desc="Extracting potential tokens", ncols=80)

    for trigram, probability in trigrams:
        pbar.update()
        for i, sequence in enumerate(tagged_sequences):
            for j in range(len(sequence) - 2):
                if (
                    all(
                        (
                            trigram[k] == sequence[j + k][0]
                            if sequence[j + k][1] == "<com>"
                            else True
                        )
                        for k in range(3)
                    )
                    and sum(sequence[j + k][1] == "<com>" for k in range(3)) == 2
                ):
                    for k in range(3):
                        if sequence[j + k][1] == "<inc>":
                            potential_tokens_from_trigrams[(i, j + k)].append(
                                (trigram[k], probability)
                            )

    for bigram, probability in bigrams:
        pbar.update()
        for i, sequence in enumerate(tagged_sequences):
            for j in range(len(sequence) - 1):
                if (
                    all(
                        (
                            bigram[k] == sequence[j + k][0]
                            if sequence[j + k][1] == "<com>"
                            else True
                        )
                        for k in range(2)
                    )
                    and sum(sequence[j + k][1] == "<com>" for k in range(2)) == 1
                ):
                    for k in range(2):
                        if sequence[j + k][1] == "<inc>":
                            potential_tokens_from_bigrams[(i, j + k)].append(
                                (bigram[k], probability)
                            )

    for unigram, probability in unigrams:
        pbar.update()
        for i, sequence in enumerate(tagged_sequences):
            for j in range(len(sequence)):
                if sequence[j][1] == "<inc>":
                    potential_tokens_from_unigrams[(i, j)].append(
                        (unigram, probability)
                    )

    pbar.close()

    return (
        dict(potential_tokens_from_unigrams),
        dict(potential_tokens_from_bigrams),
        dict(potential_tokens_from_trigrams),
    )


def min_edit_distance(source, target):
    """
    Input:
        source: token with missing characters ('-' or '…')
        target: candidate token
    Output:
        distance: minimum edit distance (float)
    """
    distance_matrix = np.zeros((len(source) + 1, len(target) + 1))

    for i in range(len(source) + 1):
        distance_matrix[i][0] = i
    for j in range(len(target) + 1):
        distance_matrix[0][j] = j

    for i in range(1, len(source) + 1):
        for j in range(1, len(target) + 1):
            if source[i - 1] == target[j - 1]:
                distance_matrix[i][j] = distance_matrix[i - 1][j - 1]
            elif source[i - 1] == "-":
                distance_matrix[i][j] = distance_matrix[i - 1][j - 1]
            elif source[i - 1] == "…":
                distance_matrix[i][j] = min(
                    distance_matrix[i - 1][j - 1], distance_matrix[i][j - 1]
                )
            else:
                distance_matrix[i][j] = min(
                    distance_matrix[i - 1][j - 1] + 2,
                    distance_matrix[i - 1][j] + 1,
                    distance_matrix[i][j - 1] + 1,
                )
    return distance_matrix[-1][-1]


def get_best_candidates(
    sequences,
    bigram_candidates_dict,
    trigram_candidates_dict,
    unigram_candidates_dict=None,
    maximum_score=0,
    number_predictions=float("inf"),
    use_unigrams=True,
):
    """
    Input:
        sequences: tagged sequences
        bigram_candidates_dict: bigram candidate dictionary
        trigram_candidates_dict: trigram candidate dictionary
        unigram_candidates_dict: unigram candidate dictionary (optional)
        maximum_score: max allowed minimum edit distance
        number_predictions: number of top predictions to keep
        use_unigrams: whether to use unigrams for <inc> inference
    Output:
        best_candidates: dictionary with best candidates per token
    """
    best_candidates = {}
    total_length = sum(
        sum(1 for token_tag in sequence if token_tag[1] == "<inc>")
        for sequence in sequences
    )
    pbar = tqdm(total=total_length, desc="Extracting best candidates", ncols=80)

    for i, sequence in enumerate(sequences):
        for j, token_tag in enumerate(sequence):
            if token_tag[1] == "<inc>":
                pbar.update()
                max_score_unigrams = []
                max_score_bigrams = []
                max_score_trigrams = []

                # Use unigrams only if enabled
                if (
                    use_unigrams
                    and unigram_candidates_dict
                    and (i, j) in unigram_candidates_dict.keys()
                ):
                    for candidate, probability in unigram_candidates_dict[i, j]:
                        med_score = min_edit_distance(token_tag[0], candidate)
                        if med_score <= maximum_score:
                            max_score_unigrams.append(
                                (candidate, med_score, probability)
                            )

                if (i, j) in bigram_candidates_dict.keys():
                    for candidate, probability in bigram_candidates_dict[i, j]:
                        med_score = min_edit_distance(token_tag[0], candidate)
                        if med_score <= maximum_score:
                            max_score_bigrams.append(
                                (candidate, med_score, probability)
                            )

                if (i, j) in trigram_candidates_dict.keys():
                    for candidate, probability in trigram_candidates_dict[i, j]:
                        med_score = min_edit_distance(token_tag[0], candidate)
                        if med_score <= maximum_score:
                            max_score_trigrams.append(
                                (candidate, med_score, probability)
                            )

                max_score_unigrams = sorted(
                    list(set(max_score_unigrams)), key=lambda x: (x[1], -x[2])
                )
                max_score_bigrams = sorted(
                    list(set(max_score_bigrams)), key=lambda x: (x[1], -x[2])
                )
                max_score_trigrams = sorted(
                    list(set(max_score_trigrams)), key=lambda x: (x[1], -x[2])
                )

                potential_candidates_trigrams = [c[0] for c in max_score_trigrams]
                potential_candidates_bigrams = [c[0] for c in max_score_bigrams]
                potential_candidates_unigrams = [c[0] for c in max_score_unigrams]

                merged_candidates = potential_candidates_trigrams + [
                    c
                    for c in potential_candidates_bigrams
                    if c not in potential_candidates_trigrams
                ]

                # Append unigrams only if enabled
                if use_unigrams:
                    merged_candidates += [
                        c
                        for c in potential_candidates_unigrams
                        if c not in potential_candidates_bigrams
                        and c not in potential_candidates_trigrams
                    ]

                best_candidates[i, j] = merged_candidates[:number_predictions]

    pbar.close()
    return best_candidates


def integrate_candidates(sequences, best_candidates, k):
    """
    Input:
        sequences: list of sequences
        best_candidates: dictionary of best candidates per token
        k: number of top candidates to include
    Output:
        sequences: sequences with inserted candidate predictions
    """
    for i, sequence in enumerate(sequences):
        for j, _ in enumerate(sequence):
            if (i, j) in best_candidates.keys():
                sequence[j] = (best_candidates[(i, j)][:k], "<mod>")
    return sequences


def extract_test_samples(test_set):
    """
    Input:
        test_set: list of test sequences
    Output:
        test_set: filtered list of usable test sequences
    Description:
        Extracts contiguous subsequences of complete (<com>) tokens.
        Returns only sequences of at least 4 complete tokens.
    """
    filtered_sequences = []

    for sequence in test_set:
        longest_sequence = []
        current_sequence = []

        for token, tag in sequence:
            if tag == "<com>":
                current_sequence.append(token)
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


def _first_candidate_for_inc(mod_sequence_entry, original_tag):
    """
    Input:
        mod_sequence_entry: either (token, '<com>/<inc>/<mis>') or ([candidates], '<mod>')
        original_tag: original tag before modification at this position
    Output:
        token_or_none: top-1 predicted token for '<inc>' if available, otherwise None
    """
    token_or_none = None
    entry_token, entry_tag = mod_sequence_entry
    if original_tag == "<com>":
        token_or_none = entry_token
    elif original_tag == "<inc>":
        if (
            entry_tag == "<mod>"
            and isinstance(entry_token, list)
            and len(entry_token) > 0
        ):
            token_or_none = entry_token[0]
    return token_or_none


def _build_ngram_indices(
    trigram_tokens: List[Tuple[Tuple[str, str, str], float]],
    bigram_tokens: Optional[List[Tuple[Tuple[str, str], float]]] = None,
    use_bigrams_for_mis: bool = True,
) -> Tuple[
    Dict[int, Dict[Tuple[str, str], List[Tuple[str, float]]]],
    Dict[int, Dict[str, List[Tuple[str, float]]]],
]:
    """
    Input:
        trigram_tokens: list of ((t1, t2, t3), prob)
        bigram_tokens: list of ((t1, t2), prob)
        use_bigrams_for_mis: whether to index bigrams
    Output:
        trigram_index_positions, bigram_index_positions
    """
    trigram_index_positions = {
        0: defaultdict(list),  # key: (t2, t3) -> list of (t1, prob)
        1: defaultdict(list),  # key: (t1, t3) -> list of (t2, prob)
        2: defaultdict(list),  # key: (t1, t2) -> list of (t3, prob)
    }
    for (t1, t2, t3), prob in trigram_tokens:
        trigram_index_positions[0][(t2, t3)].append((t1, prob))
        trigram_index_positions[1][(t1, t3)].append((t2, prob))
        trigram_index_positions[2][(t1, t2)].append((t3, prob))

    bigram_index_positions = {0: defaultdict(list), 1: defaultdict(list)}
    if use_bigrams_for_mis and bigram_tokens is not None:
        for (b1, b2), prob in bigram_tokens:
            bigram_index_positions[0][b2].append((b1, prob))
            bigram_index_positions[1][b1].append((b2, prob))

    return trigram_index_positions, bigram_index_positions


def predict_missing_candidates(
    original_tagged_sequences,
    sequences_after_inc,
    trigram_tokens,
    bigram_tokens,
    use_bigrams_for_mis=True,
    k=100,
    use_normalization_for_mis=False,
    trigram_tokens_normalized: Optional[
        List[Tuple[Tuple[str, str, str], float]]
    ] = None,
    bigram_tokens_normalized: Optional[List[Tuple[Tuple[str, str], float]]] = None,
    runic_to_normalized: Optional[Dict[str, str]] = None,
):
    """
    Input:
        original_tagged_sequences: sequences with original tags (<com>/<inc>/<mis>)
        sequences_after_inc: sequences where <inc> positions have been replaced by (<candidates>, '<mod>')
        trigram_tokens: list of ((t1, t2, t3), prob) in literal space
        bigram_tokens: list of ((t1, t2), prob) in literal space
        use_bigrams_for_mis: whether to use bigrams for <mis> inference
        k: number of top candidates to keep
        use_normalization_for_mis: enable normalized-space lookup as a back-off
        trigram_tokens_normalized: normalized-space trigrams ((n1, n2, n3), prob)
        bigram_tokens_normalized: normalized-space bigrams ((n1, n2), prob)
        runic_to_normalized: mapping from runic -> normalized
    Output:
        best_candidates_mis: dict mapping (i, j) -> ordered list of candidate tokens

    Behavior:
        - First performs literal-space search (as before).
        - If normalization is enabled, also performs normalized-space search by normalizing context tokens.
        - Merges candidates (literal priority; no duplicates).
    """
    best_candidates_mis = {}
    pbar = tqdm(
        desc="Predicting <mis> candidates",
        total=sum(len(seq) for seq in original_tagged_sequences),
        ncols=80,
    )

    # Literal indices
    tri_idx_lit, bi_idx_lit = _build_ngram_indices(
        trigram_tokens, bigram_tokens, use_bigrams_for_mis=use_bigrams_for_mis
    )

    # Normalized indices (optional)
    if use_normalization_for_mis:
        tri_idx_norm, bi_idx_norm = _build_ngram_indices(
            trigram_tokens_normalized or [],
            bigram_tokens_normalized or [],
            use_bigrams_for_mis=use_bigrams_for_mis,
        )

    def neighbor_info(seq_idx, pos):
        if pos < 0 or pos >= len(original_tagged_sequences[seq_idx]):
            return None, None
        orig_token, orig_tag = original_tagged_sequences[seq_idx][pos]
        mod_entry = sequences_after_inc[seq_idx][pos]
        if orig_tag == "<com>":
            return orig_token, "<com>"
        if orig_tag == "<inc>":
            return _first_candidate_for_inc(mod_entry, "<inc>"), "<inc>"
        return None, "<mis>"

    def unique_sorted(items: Iterable[Tuple[str, float]]) -> List[str]:
        seen = set()
        out = []
        for cand, prob in sorted(items, key=lambda x: -x[1]):
            if cand not in seen:
                seen.add(cand)
                out.append(cand)
        return out

    for i, (orig_seq, mod_seq) in enumerate(
        zip(original_tagged_sequences, sequences_after_inc)
    ):
        for j, (token, tag) in enumerate(orig_seq):
            pbar.update()
            if tag != "<mis>":
                continue

            # Literal-space candidate buckets
            candidates_by_category_lit = {
                "tri_com+com": [],
                "tri_inc+com": [],
                "tri_inc+inc": [],
                "bi_com": [],
                "bi_inc": [],
            }

            # Normalized-space candidate buckets (if enabled)
            if use_normalization_for_mis:
                candidates_by_category_norm = {
                    "tri_com+com": [],
                    "tri_inc+com": [],
                    "tri_inc+inc": [],
                    "bi_com": [],
                    "bi_inc": [],
                }

            def add_trigram(window_start, target_idx, norm_mode=False):
                pos0 = window_start
                pos1 = window_start + 1
                pos2 = window_start + 2
                if pos0 < 0 or pos2 >= len(orig_seq):
                    return

                neighbors = []
                for p in [pos0, pos1, pos2]:
                    tok, kind = neighbor_info(i, p)
                    neighbors.append((tok, kind))

                absolute_target_pos = window_start + target_idx
                if absolute_target_pos != j:
                    return

                other_positions = [0, 1, 2]
                other_positions.remove(target_idx)
                kinds = []
                toks = []
                for idx_ in other_positions:
                    tok, knd = neighbors[idx_]
                    if tok is None:
                        return
                    toks.append(tok)
                    kinds.append(knd)

                # Choose index and key map
                idx_map = {
                    0: (neighbors[1][0], neighbors[2][0]),
                    1: (neighbors[0][0], neighbors[2][0]),
                    2: (neighbors[0][0], neighbors[1][0]),
                }
                key_pair = idx_map[target_idx]

                if norm_mode:
                    if runic_to_normalized is None:
                        return
                    # Normalize key
                    nkey = (
                        normalize_token_with_map(key_pair[0], runic_to_normalized),
                        normalize_token_with_map(key_pair[1], runic_to_normalized),
                    )
                    index = (
                        tri_idx_norm[target_idx] if use_normalization_for_mis else None
                    )
                    if index is None:
                        return
                    bucket = candidates_by_category_norm
                else:
                    nkey = key_pair
                    index = tri_idx_lit[target_idx]
                    bucket = candidates_by_category_lit

                cat = None
                if kinds.count("<com>") == 2:
                    cat = "tri_com+com"
                elif kinds.count("<com>") == 1 and kinds.count("<inc>") == 1:
                    cat = "tri_inc+com"
                elif kinds.count("<inc>") == 2:
                    cat = "tri_inc+inc"

                if cat is not None:
                    for cand, prob in index.get(nkey, []):
                        bucket[cat].append((cand, prob))

            def add_bigram(window_start, target_idx, norm_mode=False):
                if not use_bigrams_for_mis:
                    return
                pos0 = window_start
                pos1 = window_start + 1
                if pos0 < 0 or pos1 >= len(orig_seq):
                    return
                neighbors = [neighbor_info(i, pos0), neighbor_info(i, pos1)]
                absolute_target_pos = window_start + target_idx
                if absolute_target_pos != j:
                    return

                other_idx = 1 - target_idx
                other_tok, other_kind = neighbors[other_idx]
                if other_tok is None:
                    return

                if norm_mode:
                    if runic_to_normalized is None:
                        return
                    other_tok_key = normalize_token_with_map(
                        other_tok, runic_to_normalized
                    )
                    index = (
                        bi_idx_norm[target_idx] if use_normalization_for_mis else None
                    )
                    if index is None:
                        return
                    bucket = candidates_by_category_norm
                else:
                    other_tok_key = other_tok
                    index = bi_idx_lit[target_idx]
                    bucket = candidates_by_category_lit

                for cand, prob in index.get(other_tok_key, []):
                    if other_kind == "<com>":
                        bucket["bi_com"].append((cand, prob))
                    elif other_kind == "<inc>":
                        bucket["bi_inc"].append((cand, prob))

            # Literal windows
            add_trigram(j - 2, 2, norm_mode=False)
            add_trigram(j - 1, 1, norm_mode=False)
            add_trigram(j, 0, norm_mode=False)
            add_bigram(j - 1, 1, norm_mode=False)
            add_bigram(j, 0, norm_mode=False)

            ordered = []
            ordered.extend(unique_sorted(candidates_by_category_lit["tri_com+com"]))
            ordered.extend(
                [
                    c
                    for c in unique_sorted(candidates_by_category_lit["tri_inc+com"])
                    if c not in ordered
                ]
            )
            ordered.extend(
                [
                    c
                    for c in unique_sorted(candidates_by_category_lit["tri_inc+inc"])
                    if c not in ordered
                ]
            )
            if use_bigrams_for_mis:
                ordered.extend(
                    [
                        c
                        for c in unique_sorted(candidates_by_category_lit["bi_com"])
                        if c not in ordered
                    ]
                )
                ordered.extend(
                    [
                        c
                        for c in unique_sorted(candidates_by_category_lit["bi_inc"])
                        if c not in ordered
                    ]
                )

            # Normalized windows (optional)
            if use_normalization_for_mis:
                add_trigram(j - 2, 2, norm_mode=True)
                add_trigram(j - 1, 1, norm_mode=True)
                add_trigram(j, 0, norm_mode=True)
                add_bigram(j - 1, 1, norm_mode=True)
                add_bigram(j, 0, norm_mode=True)

                ordered_norm = []
                ordered_norm.extend(
                    unique_sorted(candidates_by_category_norm["tri_com+com"])
                )
                ordered_norm.extend(
                    [
                        c
                        for c in unique_sorted(
                            candidates_by_category_norm["tri_inc+com"]
                        )
                        if c not in ordered_norm
                    ]
                )
                ordered_norm.extend(
                    [
                        c
                        for c in unique_sorted(
                            candidates_by_category_norm["tri_inc+inc"]
                        )
                        if c not in ordered_norm
                    ]
                )
                if use_bigrams_for_mis:
                    ordered_norm.extend(
                        [
                            c
                            for c in unique_sorted(
                                candidates_by_category_norm["bi_com"]
                            )
                            if c not in ordered_norm
                        ]
                    )
                    ordered_norm.extend(
                        [
                            c
                            for c in unique_sorted(
                                candidates_by_category_norm["bi_inc"]
                            )
                            if c not in ordered_norm
                        ]
                    )

                # Merge: keep literal priority, then add normalized suggestions
                for cand in ordered_norm:
                    if cand not in ordered:
                        ordered.append(f"{cand} (norm)")

            best_candidates_mis[(i, j)] = ordered[:k]

    pbar.close()
    return best_candidates_mis


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
) -> Dict[str, float]:
    """
    Input:
        gold_pred_list: list of tuples (gold sequence, predicted sequence)
        original_tagged_list: original tagged sequences (with <com>/<inc>/<mis>)
        runic_to_normalized: mapping from runic -> normalized
    Output:
        metrics dict with literal and normalized scores for <mis> positions

    Computes, for <mis> only:
        - prediction_coverage_literal, accuracy_literal, MRR_literal
        - prediction_coverage_norm, accuracy_norm, MRR_norm
    """
    # Literal counters
    correct_lit = 0
    incorrect_lit = 0
    nonpred_lit = 0
    rr_lit: List[float] = []

    # Normalized counters
    correct_norm = 0
    incorrect_norm = 0
    nonpred_norm = 0
    rr_norm: List[float] = []

    for idx, item in enumerate(gold_pred_list):
        gold, pred = item
        orig_tags = [tag for _, tag in original_tagged_list[idx]]

        # Pre-normalize ground truth for normalized evaluation
        gold_norm = _normalize_sequence(gold, runic_to_normalized)

        for i, predictions in enumerate(pred):
            if predictions[1] != "<mod>":
                continue
            if orig_tags[i] != "<mis>":
                continue

            candidates = predictions[0] if isinstance(predictions[0], list) else []
            # Literal eval
            if len(candidates) == 0:
                nonpred_lit += 1
            else:
                if gold[i] in candidates:
                    correct_lit += 1
                    rank_lit = candidates.index(gold[i]) + 1
                    rr_lit.append(1.0 / rank_lit)
                else:
                    incorrect_lit += 1

            # Normalized eval
            cand_norm = [
                _normalize_sequence([c], runic_to_normalized)[0] for c in candidates
            ]
            if len(cand_norm) == 0:
                nonpred_norm += 1
            else:
                if gold_norm[i] in cand_norm:
                    correct_norm += 1
                    rank_norm = cand_norm.index(gold_norm[i]) + 1
                    rr_norm.append(1.0 / rank_norm)
                else:
                    incorrect_norm += 1

    # Aggregate literal
    pred_lit = correct_lit + incorrect_lit
    to_pred_lit = pred_lit + nonpred_lit
    coverage_lit = (to_pred_lit - nonpred_lit) / to_pred_lit if to_pred_lit else 0.0
    acc_lit = correct_lit / pred_lit if pred_lit else 0.0
    mrr_lit = sum(rr_lit) / len(rr_lit) if rr_lit else 0.0

    # Aggregate normalized
    pred_norm = correct_norm + incorrect_norm
    to_pred_norm = pred_norm + nonpred_norm
    coverage_norm = (
        (to_pred_norm - nonpred_norm) / to_pred_norm if to_pred_norm else 0.0
    )
    acc_norm = correct_norm / pred_norm if pred_norm else 0.0
    mrr_norm = sum(rr_norm) / len(rr_norm) if rr_norm else 0.0

    return {
        "prediction_coverage_literal": coverage_lit,
        "accuracy_literal": acc_lit,
        "MRR_literal": mrr_lit,
        "prediction_coverage_norm": coverage_norm,
        "accuracy_norm": acc_norm,
        "MRR_norm": mrr_norm,
    }


# Normalization utilities

def load_normalization_mapping(json_path: Union[str, Path]) -> Dict[str, str]:
    """
    Input:
        json_path: path to scandi_runic_all_mapping.json
    Output:
        runic_to_normalized: dict mapping runic variant -> normalized Old Scandinavian form

    Notes:
        The JSON file is expected to have the structure:
        {
          "<NormalizedName>": {
            "<runic_variant_1>": {"signatures": [...], "count": n},
            "<runic_variant_2>": {"signatures": [...], "count": m},
            ...
          },
          ...
        }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    runic_to_normalized: Dict[str, str] = {}
    for normalized, variants in mapping.items():
        for runic_variant in variants.keys():
            runic_to_normalized[runic_variant] = normalized
    return runic_to_normalized


def normalize_token_with_map(token: str, runic_to_normalized: Dict[str, str]) -> str:
    """
    Input:
        token: runic token to normalize
        runic_to_normalized: mapping from runic variants to normalized name
    Output:
        normalized_token: normalized form if available, else original token
    """
    if not isinstance(token, str):
        return token
    return runic_to_normalized.get(token, token)


def normalize_ngram_probabilities(
    unigrams: Dict[str, float],
    bigrams: Dict[str, Dict[str, float]],
    trigrams: Dict[Tuple[str, str], Dict[str, float]],
    runic_to_normalized: Dict[str, str],
) -> Tuple[
    Dict[str, float],
    Dict[str, Dict[str, float]],
    Dict[Tuple[str, str], Dict[str, float]],
]:
    """
    Input:
        unigrams, bigrams, trigrams: literal probability dictionaries
        runic_to_normalized: mapping from runic variant -> normalized name
    Output:
        norm_unigrams, norm_bigrams, norm_trigrams: probability dictionaries in normalized space

    Behavior:
        - Replace every token by its normalized equivalent (if known).
        - Merge duplicate entries by summing probabilities.
        - Renormalize within each context so probabilities remain valid (sum to 1).
    """
    # Unigrams
    norm_unigrams_counter: Dict[str, float] = defaultdict(float)
    for tok, prob in unigrams.items():
        ntok = normalize_token_with_map(tok, runic_to_normalized)
        norm_unigrams_counter[ntok] += float(prob)
    total_u = sum(norm_unigrams_counter.values()) or 1.0
    norm_unigrams = {tok: prob / total_u for tok, prob in norm_unigrams_counter.items()}

    # Bigrams
    norm_bigrams_counter: Dict[str, Counter] = defaultdict(Counter)
    for left, right_probs in bigrams.items():
        nleft = normalize_token_with_map(left, runic_to_normalized)
        merge_counter: Counter = Counter()
        for right, prob in right_probs.items():
            nright = normalize_token_with_map(right, runic_to_normalized)
            merge_counter[nright] += float(prob)
        # Renormalize for this left context
        total_b = sum(merge_counter.values()) or 1.0
        norm_bigrams_counter[nleft] = Counter(
            {r: p / total_b for r, p in merge_counter.items()}
        )
    norm_bigrams = {l: dict(c) for l, c in norm_bigrams_counter.items()}

    # Trigrams
    norm_trigrams_counter: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for (w1, w2), right_probs in trigrams.items():
        nw1 = normalize_token_with_map(w1, runic_to_normalized)
        nw2 = normalize_token_with_map(w2, runic_to_normalized)
        merge_counter = Counter()
        for w3, prob in right_probs.items():
            nw3 = normalize_token_with_map(w3, runic_to_normalized)
            merge_counter[nw3] += float(prob)
        total_t = sum(merge_counter.values()) or 1.0
        norm_trigrams_counter[(nw1, nw2)] = Counter(
            {r: p / total_t for r, p in merge_counter.items()}
        )
    norm_trigrams = {k: dict(c) for k, c in norm_trigrams_counter.items()}

    return norm_unigrams, norm_bigrams, norm_trigrams


def extract_unigram_tokens_from_probs(
    unigrams_dictionary: Dict[str, float],
) -> List[Tuple[str, float]]:
    """
    Input:
        unigrams_dictionary: dictionary of unigrams and probabilities
    Output:
        unigram_words: list of (unigram, probability) tuples
    """
    return list(
        {
            (word, float(probability))
            for word, probability in unigrams_dictionary.items()
        }
    )


def extract_bigram_tokens_from_probs(
    bigrams_dictionary: Dict[str, Dict[str, float]],
) -> List[Tuple[Tuple[str, str], float]]:
    """
    Input:
        bigrams_dictionary: dictionary of bigrams and probabilities (left -> {right: prob})
    Output:
        bigram_words: list of ((token1, token2), probability) tuples
    """
    bigram_words: List[Tuple[Tuple[str, str], float]] = []
    for left, right_dict in bigrams_dictionary.items():
        for right, prob in right_dict.items():
            bigram_words.append(((left, right), float(prob)))
    # Deduplicate
    return list({(pair, p) for pair, p in bigram_words})


def extract_trigram_tokens_from_probs(
    trigrams_dictionary: Dict[Tuple[str, str], Dict[str, float]],
) -> List[Tuple[Tuple[str, str, str], float]]:
    """
    Input:
        trigrams_dictionary: dictionary of trigrams and probabilities ((t1, t2) -> {t3: prob})
    Output:
        trigram_words: list of ((token1, token2, token3), probability) tuples
    """
    trigram_words: List[Tuple[Tuple[str, str, str], float]] = []
    for (t1, t2), right_dict in trigrams_dictionary.items():
        for t3, prob in right_dict.items():
            trigram_words.append(((t1, t2, t3), float(prob)))
    return list({(trip, p) for trip, p in trigram_words})


def save_normalized_ngram_pickles(
    norm_unigrams: Dict[str, float],
    norm_bigrams: Dict[str, Dict[str, float]],
    norm_trigrams: Dict[Tuple[str, str], Dict[str, float]],
) -> None:
    """
    Input:
        norm_unigrams, norm_bigrams, norm_trigrams: normalized probability dicts
    Output:
        None (writes *_normalized.pkl files to disk)
    """
    uni = extract_unigram_tokens_from_probs(norm_unigrams)
    bi = extract_bigram_tokens_from_probs(norm_bigrams)
    tri = extract_trigram_tokens_from_probs(norm_trigrams)

    with open("unigram_tokens_normalized.pkl", "wb") as f:
        pickle.dump(uni, f)
    with open("bigram_tokens_normalized.pkl", "wb") as f:
        pickle.dump(bi, f)
    with open("trigram_tokens_normalized.pkl", "wb") as f:
        pickle.dump(tri, f)


# Main processing function

def main(
    test_mode=False,
    mask_missing_during_eval=True,
    mask_fraction=1 / 6,
    use_bigrams_for_mis=True,
    use_normalization_for_mis=True,
    use_unigrams=True,
    normalization_json_path: Union[str, Path] = "scandi_runic_all_mapping.json",
):
    """
    Input:
        test_mode: boolean, if True, uses a reduced dataset for faster testing
        mask_missing_during_eval: whether to create <mis> tokens before <inc> noise in evaluation data
        mask_fraction: fraction of tokens to mask as <mis> during evaluation when enabled
        use_bigrams_for_mis: whether bigrams are used for <mis> prediction (after trigrams)
        use_normalization_for_mis: enable normalized back-off for <mis> prediction
        use_unigrams: whether to include unigrams for <inc> candidate generation
        normalization_json_path: path to scandi_runic_all_mapping.json
    Output:
        - Processed CSV, pickle files, and n-gram probability text file
        - Normalized n-gram pickle files
        - Excel files with evaluation metrics for <inc>, <mis> (literal + normalized), and combined
        - predictions_k100.txt with top-100 candidate outputs
    """
    print("=== Runic N-gram Processing Script ===")
    print("Mode:", "TEST" if test_mode else "FULL")
    print(
        f"Mask <mis> during eval: {mask_missing_during_eval} (fraction={mask_fraction})"
    )
    print(f"Use bigrams for <mis>: {use_bigrams_for_mis}")
    print(f"Use normalization for <mis>: {use_normalization_for_mis}")
    print(f"Use unigrams for <inc>: {use_unigrams}")
    print()

    # Step 0: Load normalization mapping (if available / desired)
    runic_to_normalized: Dict[str, str] = {}
    if use_normalization_for_mis:
        try:
            runic_to_normalized = load_normalization_mapping(normalization_json_path)
            print(
                f"Loaded normalization mapping from '{normalization_json_path}' "
                f"({len(runic_to_normalized)} runic variants)."
            )
        except FileNotFoundError:
            print(
                f"Warning: '{normalization_json_path}' not found. Continuing without normalization."
            )
            use_normalization_for_mis = False

    # Step 1: Read Excel file
    df = read_excel_to_dataframe("rundata-net_results.xlsx")

    # Step 2: Limit size in test mode only
    if test_mode:
        df = df.head(200)
        print(f"Running in test mode with {len(df)} inscriptions (subset).")

    # Step 3–7: Clean, tokenize, tag, and filter
    runestones = df["Transliterated runic text"].dropna().tolist()
    df_runestones = pd.DataFrame(runestones, columns=["Transliterated runic text"])
    punct = print_non_alpha(runestones)
    df_runestones["Transliterated runic text"] = df_runestones[
        "Transliterated runic text"
    ].apply(lambda x: remove_punctuation(x, punct))
    df_runestones["Transliterated runic text"] = df_runestones[
        "Transliterated runic text"
    ].apply(tokenize_text)
    df_runestones["Transliterated runic text"] = df_runestones[
        "Transliterated runic text"
    ].apply(get_tags)
    df_runestones = df_runestones[
        df_runestones["Transliterated runic text"].apply(lambda x: len(x) >= 3)
    ]
    df_runestones.to_csv("Processed_Runestones.csv", index=False)

    # Step 8–10: Train/test split
    runestones = df_runestones["Transliterated runic text"].tolist()
    train, test = train_test_split(runestones, test_size=0.05, random_state=27)
    print(f"Training set size: {len(train)}")
    print(f"Test set size: {len(test)}")

    # Helper to strip tags
    def strip_tags(sequence_list):
        return [" ".join(token for token, _ in seq) for seq in sequence_list]

    # Step 11: Save plain text sequences
    with open("training_sequences.txt", "w", encoding="utf-8") as f_train:
        f_train.writelines(seq + "\n" for seq in strip_tags(train))
    with open("test_sequences.txt", "w", encoding="utf-8") as f_test:
        f_test.writelines(seq + "\n" for seq in strip_tags(test))
    print("Training and test sequences saved to text files.")

    # Step 12: Compute and save n-gram probabilities (literal)
    runic_unigrams, runic_bigrams, runic_trigrams = extract_ngram_probabilities(train)
    save_probabilities_to_file(
        runic_unigrams, runic_bigrams, runic_trigrams, "n-gram_probabilities.txt"
    )

    # Step 12b: Build normalized probability dicts and pickles (if enabled)
    if use_normalization_for_mis and runic_to_normalized:
        n_uni, n_bi, n_tri = normalize_ngram_probabilities(
            runic_unigrams, runic_bigrams, runic_trigrams, runic_to_normalized
        )
    else:
        n_uni, n_bi, n_tri = {}, {}, {}

    # Step 13: Extract and pickle token data (literal)
    extracted_unigram_tokens = extract_unigram_tokens(runic_unigrams)
    extracted_bigram_tokens = extract_bigram_tokens(runic_bigrams)
    extracted_trigram_tokens = extract_trigram_tokens(runic_trigrams)
    for name, data in [
        ("unigram_tokens.pkl", extracted_unigram_tokens),
        ("bigram_tokens.pkl", extracted_bigram_tokens),
        ("trigram_tokens.pkl", extracted_trigram_tokens),
    ]:
        with open(name, "wb") as f:
            pickle.dump(data, f)

    # Step 13b: Extract and pickle token data (normalized)
    if use_normalization_for_mis and n_uni and n_bi and n_tri:
        save_normalized_ngram_pickles(n_uni, n_bi, n_tri)
        with open("unigram_tokens_normalized.pkl", "rb") as f:
            extracted_unigram_tokens_norm = pickle.load(f)
        with open("bigram_tokens_normalized.pkl", "rb") as f:
            extracted_bigram_tokens_norm = pickle.load(f)
        with open("trigram_tokens_normalized.pkl", "rb") as f:
            extracted_trigram_tokens_norm = pickle.load(f)
    else:
        extracted_unigram_tokens_norm = []
        extracted_bigram_tokens_norm = []
        extracted_trigram_tokens_norm = []

    # Step 14: Prepare test data
    test_keys = extract_test_samples(test)
    print(f"Appropriate test sequences: {len(test_keys)}")

    test_keys_copy = [seq for seq in test_keys for _ in range(5)]
    test_exercise = alter_sequences(
        test_keys, mask_missing=mask_missing_during_eval, mask_fraction=mask_fraction
    )
    print(f"Synthetic dataset size: {len(test_exercise)}")

    test_exercise_tagged = [get_tags(seq) for seq in test_exercise]
    with open("synthetic_test_sequences.txt", "w", encoding="utf-8") as f_syn:
        f_syn.writelines(" ".join(seq) + "\n" for seq in test_exercise)
    print("Synthetic test sequences saved to 'synthetic_test_sequences.txt'.")

    # Step 15: Candidate predictions for <inc>
    unigram_candidates_dict, bigram_candidates_dict, trigram_candidates_dict = (
        extract_potential_tokens(
            test_exercise_tagged,
            extracted_unigram_tokens,
            extracted_bigram_tokens,
            extracted_trigram_tokens,
        )
    )

    best_candidates_inc = get_best_candidates(
        test_exercise_tagged,
        bigram_candidates_dict,
        trigram_candidates_dict,
        unigram_candidates_dict=unigram_candidates_dict if use_unigrams else None,
        maximum_score=0,
        number_predictions=100,
        use_unigrams=use_unigrams,
    )

    # Step 16: Evaluation setup
    results_inc = pd.DataFrame(columns=["k", "prediction_coverage", "accuracy", "MRR"])
    results_mis = pd.DataFrame(
        columns=[
            "k",
            "prediction_coverage_literal",
            "accuracy_literal",
            "MRR_literal",
            "prediction_coverage_norm",
            "accuracy_norm",
            "MRR_norm",
        ]
    )
    results_both = pd.DataFrame(columns=["k", "prediction_coverage", "accuracy", "MRR"])
    k_values = range(10, 101, 10)
    predictions_file = "predictions_k100.txt"
    example_counter = 0

    # Clear prediction file
    with open(predictions_file, "w", encoding="utf-8") as f_out:
        f_out.write("=== Predictions at k=100 ===\n")

    # Step 17: Evaluation loop
    for k in k_values:
        # 17a: insert <inc> predictions
        seq_after_inc = integrate_candidates(
            copy.deepcopy(test_exercise_tagged), best_candidates_inc, k
        )

        # 17b: predict <mis> using trigrams (and optionally bigrams), with optional normalization
        best_candidates_mis = predict_missing_candidates(
            original_tagged_sequences=test_exercise_tagged,
            sequences_after_inc=seq_after_inc,
            trigram_tokens=extracted_trigram_tokens,
            bigram_tokens=extracted_bigram_tokens,
            use_bigrams_for_mis=use_bigrams_for_mis,
            k=k,
            use_normalization_for_mis=use_normalization_for_mis,
            trigram_tokens_normalized=(
                extracted_trigram_tokens_norm if use_normalization_for_mis else None
            ),
            bigram_tokens_normalized=(
                extracted_bigram_tokens_norm if use_normalization_for_mis else None
            ),
            runic_to_normalized=(
                runic_to_normalized if use_normalization_for_mis else None
            ),
        )

        # 17c: insert <mis> predictions
        test_predictions = integrate_candidates(seq_after_inc, best_candidates_mis, k)

        # 17d: Combine for evaluation
        zipped = zip(test_keys_copy, test_predictions, test_exercise)
        gold_pred = [
            (gold_seq, pred_seq, altered_seq)
            for gold_seq, pred_seq, altered_seq in zipped
        ]

        # Evaluate <inc> and combined
        eval_modes = ["inc", "both"]
        metrics = {}
        for mode in eval_modes:
            (
                prediction_coverage,
                accuracy,
                MRR,
                _correct_predictions,
                _incorrect_predictions,
                _non_predictions,
            ) = calculate_metrics(
                [(g, p) for g, p, _ in gold_pred],
                test_exercise_tagged,
                eval_scope=mode,
            )
            metrics[mode] = {
                "k": k,
                "prediction_coverage": prediction_coverage,
                "accuracy": accuracy,
                "MRR": MRR,
            }

        results_inc = pd.concat(
            [results_inc, pd.DataFrame([metrics["inc"]])], ignore_index=True
        )
        results_both = pd.concat(
            [results_both, pd.DataFrame([metrics["both"]])], ignore_index=True
        )

        # Evaluate <mis>: literal + normalized
        mis_metrics = calculate_mis_metrics_with_normalization(
            [(g, p) for g, p, _ in gold_pred],
            test_exercise_tagged,
            runic_to_normalized if use_normalization_for_mis else {},
        )
        mis_metrics["k"] = k
        results_mis = pd.concat(
            [results_mis, pd.DataFrame([mis_metrics])], ignore_index=True
        )

        # Save predictions only for k = 100
        if k == 100:
            with open(predictions_file, "a", encoding="utf-8") as f_out:
                for idx, (gold_seq, pred_seq, altered_seq) in enumerate(gold_pred):
                    prediction_tokens = []
                    for token_entry in pred_seq:
                        token, pt_tag = token_entry
                        if pt_tag == "<mod>":
                            if isinstance(token, list) and token:
                                prediction_tokens.append("[" + ", ".join(token) + "]")
                            else:
                                prediction_tokens.append("[]")
                        else:
                            prediction_tokens.append(token)

                    f_out.write("\n")
                    f_out.write("ground truth: " + " ".join(gold_seq) + "\n")
                    f_out.write("altered sequence: " + " ".join(altered_seq) + "\n")
                    f_out.write("prediction: " + " ".join(prediction_tokens) + "\n")

        # Print 10 examples only for k = 10
        if k == 10 and example_counter < 10:
            for idx, (gold_seq, pred_seq, altered_seq) in enumerate(gold_pred):
                if example_counter >= 10:
                    break
                prediction_tokens = []
                for token_entry in pred_seq:
                    token, pt_tag = token_entry
                    if pt_tag == "<mod>":
                        if isinstance(token, list) and token:
                            prediction_tokens.append("[" + ", ".join(token) + "]")
                        else:
                            prediction_tokens.append("[]")
                    else:
                        prediction_tokens.append(token)
                print("\nExample Prediction", example_counter + 1)
                print("ground truth:", " ".join(gold_seq))
                print("altered sequence:", " ".join(altered_seq))
                print("prediction:", " ".join(prediction_tokens))
                example_counter += 1

    # Step 18: Save results
    results_inc.to_excel("results_inc.xlsx", index=False)
    results_mis.to_excel("results_mis.xlsx", index=False)
    results_both.to_excel("results_combined.xlsx", index=False)
    print("\nProcessing complete.")
    print(
        "Saved metrics to 'results_inc.xlsx', 'results_mis.xlsx', and 'results_combined.xlsx'."
    )
    print(f"All predictions (k=100) saved to '{predictions_file}'.")


if __name__ == "__main__":
    main(
        test_mode=False,
        mask_missing_during_eval=True,
        mask_fraction=1 / 6,
        use_bigrams_for_mis=True,
        use_normalization_for_mis=True,
        use_unigrams=True,
        normalization_json_path="scandi_runic_all_mapping.json",
    )

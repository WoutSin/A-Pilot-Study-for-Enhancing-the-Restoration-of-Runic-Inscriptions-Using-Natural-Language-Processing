"""Candidates functions."""

import numpy as np
from collections import defaultdict
from tqdm import tqdm
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from .normalization import normalize_token_with_map


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

"""Normalization functions."""

import json
import pickle
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union


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
    output_dir: Path,
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

    with open(output_dir / "unigram_tokens_normalized.pkl", "wb") as f:
        pickle.dump(uni, f)
    with open(output_dir / "bigram_tokens_normalized.pkl", "wb") as f:
        pickle.dump(bi, f)
    with open(output_dir / "trigram_tokens_normalized.pkl", "wb") as f:
        pickle.dump(tri, f)

"""Ngrams functions."""

from collections import Counter
from collections import defaultdict
from itertools import product
from .preprocessing import get_tags, get_token_alternatives


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

        for token, tag in tagged_sequence:
            alternatives = get_token_alternatives(token)
            tagged_alternatives = get_tags(alternatives)

            complete_alternatives = [
                reading
                for reading, reading_tag in tagged_alternatives
                if reading_tag == "<com>"
            ]

            com_items.append(complete_alternatives)

            for reading in complete_alternatives:
                unigrams[reading] += 1

        for i in range(len(com_items) - 1):
            for word1, word2 in product(com_items[i], com_items[i + 1]):
                bigrams[word1][word2] += 1

        for i in range(len(com_items) - 2):
            for word1, word2, word3 in product(
                com_items[i], com_items[i + 1], com_items[i + 2]
            ):
                trigrams[(word1, word2)][word3] += 1

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

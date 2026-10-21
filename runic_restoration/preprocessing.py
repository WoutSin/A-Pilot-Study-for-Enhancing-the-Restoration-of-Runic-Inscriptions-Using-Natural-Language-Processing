"""Preprocessing functions."""

import pandas as pd
from sklearn.model_selection import train_test_split


def read_excel_to_dataframe(file_path):
    """
    Input:
        file_path: path to an Excel file containing transliterated runic inscriptions
    Output:
        df: dataframe containing transliterated runic texts
    """
    df = pd.read_excel(file_path, engine="openpyxl")
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
                if not char.isalnum() and char not in ["-", "…", " ", "/"]
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


def get_token_alternatives(token):
    """
    Input:
        token: token that may contain alternative readings
    Output:
        alternatives: list of possible readings of the token
    """
    alternatives = token.split("/")
    return alternatives

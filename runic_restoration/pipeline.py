"""Pipeline functions."""

import copy
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict
from typing import Union
from .candidates import extract_potential_tokens, get_best_candidates, integrate_candidates, predict_missing_candidates
from .evaluation import alter_sequences, calculate_metrics, calculate_mis_metrics_with_normalization, extract_test_samples
from .ngrams import extract_bigram_tokens, extract_ngram_probabilities, extract_trigram_tokens, extract_unigram_tokens, save_probabilities_to_file
from .normalization import load_normalization_mapping, normalize_ngram_probabilities, save_normalized_ngram_pickles
from .preprocessing import get_tags, print_non_alpha, read_excel_to_dataframe, remove_punctuation, tokenize_text


def main(
    test_mode=False,
    mask_missing_during_eval=True,
    mask_fraction=1 / 6,
    use_bigrams_for_mis=True,
    use_normalization_for_mis=True,
    use_unigrams=True,
    normalization_json_path: Union[str, Path] = "inputs/scandi_runic_all_mapping.json",
    input_path: Union[str, Path] = "inputs/rundata-net_results.xlsx",
    output_dir: Union[str, Path] = "outputs",
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
        - Excel files with evaluation metrics for <inc>, <mis> (selected mode only), and combined
        - predictions_k100.txt with top-100 candidate outputs
    """
    output_dir = Path(output_dir)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
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
    df = read_excel_to_dataframe(input_path)

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
    df_runestones.to_csv(output_dir / "Processed_Runestones.csv", index=False)

    # Step 8–10: Train/test split
    runestones = df_runestones["Transliterated runic text"].tolist()
    train, test = train_test_split(runestones, test_size=0.05, random_state=27)
    print(f"Training set size: {len(train)}")
    print(f"Test set size: {len(test)}")

    # Helper to strip tags
    def strip_tags(sequence_list):
        return [" ".join(token for token, _ in seq) for seq in sequence_list]

    # Step 11: Save plain text sequences
    with open(output_dir / "training_sequences.txt", "w", encoding="utf-8") as f_train:
        f_train.writelines(seq + "\n" for seq in strip_tags(train))
    with open(output_dir / "test_sequences.txt", "w", encoding="utf-8") as f_test:
        f_test.writelines(seq + "\n" for seq in strip_tags(test))
    print("Training and test sequences saved to text files.")

    # Step 12: Compute and save n-gram probabilities (literal)
    runic_unigrams, runic_bigrams, runic_trigrams = extract_ngram_probabilities(train)
    save_probabilities_to_file(
        runic_unigrams, runic_bigrams, runic_trigrams, output_dir / "n-gram_probabilities.txt"
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
        (models_dir / "unigram_tokens.pkl", extracted_unigram_tokens),
        (models_dir / "bigram_tokens.pkl", extracted_bigram_tokens),
        (models_dir / "trigram_tokens.pkl", extracted_trigram_tokens),
    ]:
        with open(name, "wb") as f:
            pickle.dump(data, f)

    # Step 13b: Extract and pickle token data (normalized)
    if use_normalization_for_mis and n_uni and n_bi and n_tri:
        save_normalized_ngram_pickles(n_uni, n_bi, n_tri, models_dir)
        with open(models_dir / "unigram_tokens_normalized.pkl", "rb") as f:
            extracted_unigram_tokens_norm = pickle.load(f)
        with open(models_dir / "bigram_tokens_normalized.pkl", "rb") as f:
            extracted_bigram_tokens_norm = pickle.load(f)
        with open(models_dir / "trigram_tokens_normalized.pkl", "rb") as f:
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
    with open(output_dir / "synthetic_test_sequences.txt", "w", encoding="utf-8") as f_syn:
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
    predictions_file = output_dir / "predictions_k100.txt"
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

        # Evaluate <mis> with the selected normalization setting
        mis_metrics = calculate_mis_metrics_with_normalization(
            [(g, p) for g, p, _ in gold_pred],
            test_exercise_tagged,
            runic_to_normalized if use_normalization_for_mis else {},
            use_normalization=use_normalization_for_mis,
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
    results_inc.to_excel(output_dir / "results_inc.xlsx", index=False)
    results_mis.to_excel(output_dir / "results_mis.xlsx", index=False)
    results_both.to_excel(output_dir / "results_combined.xlsx", index=False)
    print("\nProcessing complete.")
    print(
        "Saved metrics to 'results_inc.xlsx', 'results_mis.xlsx', and 'results_combined.xlsx'."
    )
    print(f"All predictions (k=100) saved to '{predictions_file}'.")

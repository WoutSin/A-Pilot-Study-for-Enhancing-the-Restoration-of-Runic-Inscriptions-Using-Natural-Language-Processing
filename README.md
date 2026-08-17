# A Pilot Study for Enhancing the Restoration of Runic Inscriptions Using Natural Language Processing Techniques

This Python script is designed to reconstruct incomplete or missing runic inscriptions from the Medieval and Viking Age periods using n-gram probabilities, a modified Minimum Edit Distance algorithm and an optional normalization layer that maps runic variants to standardized Old Scandinavian forms. The script processes a dataset of runic inscriptions, extracts n-gram probabilities and generates potential candidates for missing or incomplete tokens in the inscriptions.

## Features

- Reads transliterated runic inscriptions from an Excel file.  
- Cleans and tokenizes inscriptions and tags tokens as complete (`<com>`), incomplete (`<inc>`), or missing (`<mis>`).  
- Extracts unigram, bigram and trigram probabilities from the training data.  
- Optionally normalizes runic variants using a mapping file (`scandi_runic_all_mapping.json`) to group equivalent forms.  
- Generates potential candidates for incomplete tokens based on n-gram probabilities and contextual information.  
- Predicts missing tokens using trigram and optionally bigram probabilities, with optional normalization-based back-off.  
- Filters out the most likely candidates using a modified MED algorithm.  
- Evaluates the performance of the reconstruction using prediction coverage, accuracy and Mean Reciprocal Rank metrics.  

## Usage

1. Ensure that you have the required dependencies installed (e.g., pandas, numpy, tqdm, sklearn).  
2. Place the transliterated inscription Excel file (`rundata-net_results.xlsx`) and, if desired, the normalization mapping file (`scandi_runic_all_mapping.json`) in the same directory as the script.  
3. Run the script: `python script.py`  
4. The script will process the data, extract literal and optionally normalized n-gram probabilities, generate candidates for a synthetic test set and evaluate the performance for different values of k (maximum number of candidates considered).  
5. The results will be saved as Excel files (`results_inc.xlsx`, `results_mis.xlsx`, `results_combined.xlsx`) in the same directory, together with supporting text and pickle files.  

## Dependencies

- numpy==2.3.5
- pandas==2.3.3
- scikit-learn==1.7.2
- tqdm==4.67.1


## Hyperparameters

The script includes the following hyperparameters that can be adjusted:

- `test_mode`: If set to `True`, uses a reduced subset of the dataset for troubleshooting.  
- `mask_missing_during_eval`: If `True`, synthetic missing tokens (`<mis>`) are created before incomplete ones (`<inc>`) for evaluation.  
- `mask_fraction`: The fraction of tokens to be masked as `<mis>` in the synthetic test data.  
- `use_bigrams_for_mis`: Determines whether bigrams are used, in addition to trigrams, when predicting missing tokens.  
- `use_normalization_for_mis`: Enables normalization-based back-off during missing token prediction.  
- `use_unigrams`: Determines whether unigrams are included when generating candidates for incomplete tokens.  
- `normalization_json_path`: The path to the normalization mapping file used for variant standardization.  

## Notes

- The script automatically builds both literal and normalized n-gram probability models if a normalization mapping is available.  
- It generates a synthetic test set by altering complete inscriptions from the original dataset. The test set is used for evaluation purposes.  
- The script saves extracted n-gram probabilities to a text file (`n-gram_probabilities.txt`) and the n-gram tokens to pickle files (`unigram_tokens.pkl`, `bigram_tokens.pkl`, `trigram_tokens.pkl`), along with normalized versions when applicable.  
- Evaluation metrics are calculated separately for incomplete (`<inc>`), missing (`<mis>`) and combined predictions.  

## Notebook

The tool itself is made available as a Colab Notebook and can be found in the `Colab_Notebook` folder along with the dependencies. You can run the model on multiple inscriptions at a time by creating a file named `runic_inscriptions.txt,` providing one inscriptions per line. Be sure to employ `-` and `…` characters as placeholders.

# Runic Inscription Restoration

An experimental natural language processing (NLP) pipeline for restoring incomplete and missing text in transliterated runic inscriptions using statistical n-gram language models.

The project uses unigram, bigram, and trigram statistics to generate and rank candidate reconstructions. It also supports normalization of runic variants and evaluates restoration performance using synthetically corrupted inscriptions.

### Directory Descriptions

- **`inputs/`** — Source datasets, normalization mappings, and example inscriptions.
- **`runic_restoration/`** — Python package containing the restoration and evaluation logic.
- **`notebooks/`** — Jupyter notebooks for interactive exploration and inference.
- **`outputs/`** — Generated datasets, predictions, evaluation results, and model files.

### Python Modules

| Module | Description |
|---|---|
| `preprocessing.py` | Dataset loading, cleaning, tokenization, and tagging |
| `ngrams.py` | Construction and export of n-gram frequency and probability models |
| `candidates.py` | Candidate generation, ranking, and restoration |
| `normalization.py` | Runic variant normalization and normalized language models |
| `evaluation.py` | Synthetic corruption and evaluation metrics |
| `pipeline.py` | Execution of the complete experimental workflow |

## Installation

### Requirements

- Python 3
- pip
- GNU Make (optional)

Python package dependencies are listed in `requirements.txt`.

### Set Up the Environment

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
make install
```

Alternatively:

```bash
python3 -m pip install -r requirements.txt
```

## Usage

### Run the Complete Pipeline

Execute:

```bash
make run
```

Alternatively:

```bash
python3 run.py
```

The pipeline loads the input dataset and normalization mapping, processes the inscriptions, constructs n-gram models, generates restoration candidates, evaluates the predictions, and saves the results to `outputs/`.

### Test Mode

For a smaller experimental run:

```bash
make test
```

Alternatively:

```bash
python3 run.py --test-mode
```

### Custom Configuration

The command-line interface supports specifying input files and output directories:

```bash
python3 run.py \
    --input inputs/rundata-net_results.xlsx \
    --mapping inputs/scandi_runic_all_mapping.json \
    --output-dir outputs
```

To view all available options:

```bash
python3 run.py --help
```

## Makefile Commands

| Command | Description |
|---|---|
| `make install` | Install Python dependencies |
| `make run` | Execute the complete restoration pipeline |
| `make test` | Execute the pipeline in test mode |
| `make check` | Check Python source compilation |
| `make clean` | Remove generated reports and intermediate files while retaining model pickle files |

## Methodology

### 1. Data Preprocessing

Runic inscriptions are loaded from the input dataset and prepared for language modeling.

Processing includes text cleaning, tokenization, and identification of tokens marked as incomplete or missing.

The pipeline distinguishes between two restoration tasks:

- `<inc>` — Incomplete tokens requiring reconstruction.
- `<mis>` — Missing tokens requiring prediction.

### 2. N-gram Language Modeling

The pipeline constructs statistical language models using token sequences extracted from the inscriptions.

Three n-gram orders are supported:

- **Unigrams:** Individual token frequencies.
- **Bigrams:** Frequencies and probabilities of consecutive token pairs.
- **Trigrams:** Frequencies and probabilities of consecutive token triples.

These statistics provide contextual information for ranking potential restorations.

### 3. Candidate Generation and Restoration

Restoration candidates are generated and ranked using the available n-gram statistics.

The pipeline supports unigram-based restoration for incomplete tokens and context-based prediction for missing tokens.

### 4. Normalization

A normalization mapping is used to associate variant runic forms with standardized representations.

Normalized language models provide an additional source of information when selecting restoration candidates.

### 5. Evaluation

Restoration performance is evaluated using synthetically corrupted inscription data.

The evaluation workflow masks or alters selected tokens and compares the predicted restorations against the original reference tokens.

Results are produced separately for incomplete-token restoration, missing-token prediction, and combined evaluation.

## Outputs

Generated files are saved in `outputs/`.

These include processed inscription datasets, training and test sequences, n-gram statistics, restoration predictions, and evaluation reports.

### Evaluation Reports

| File | Description |
|---|---|
| `results_inc.xlsx` | Incomplete-token restoration evaluation |
| `results_mis.xlsx` | Missing-token prediction evaluation |
| `results_combined.xlsx` | Combined restoration evaluation |
| `Processed_Runestones.csv` | Processed runic inscription dataset |

### Language Models

Model files are stored in:

```text
outputs/models/
```

The directory contains serialized n-gram models, including literal and normalized unigram, bigram, and trigram statistics.

## Notebook

The repository includes an interactive Jupyter notebook for experimenting with the model and performing inference:

`notebooks/runic_restoration_model.ipynb`

The notebook uses serialized n-gram models and sample inscriptions to explore runic restoration behavior.

### Run in Google Colab

The notebook can be opened and executed directly in Google Colab without downloading or installing anything.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xh4xN7L5npvtv3EDQH9sZj8HVqRQLMhf?usp=sharing)

Simply click the badge, sign in with your Google account, and run the notebook.

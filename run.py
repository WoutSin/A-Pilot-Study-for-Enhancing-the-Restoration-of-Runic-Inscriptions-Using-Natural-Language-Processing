"""Command-line entry point for the runic restoration experiment."""

import argparse
from pathlib import Path

from runic_restoration.pipeline import main


def parse_args():
    """Parse experiment configuration from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("inputs/rundata-net_results.xlsx"))
    parser.add_argument("--mapping", type=Path, default=Path("inputs/scandi_runic_all_mapping.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--mask-fraction", type=float, default=1 / 6)
    parser.add_argument("--no-normalization", action="store_true")
    parser.add_argument("--no-bigrams", action="store_true")
    parser.add_argument("--no-unigrams", action="store_true")
    parser.add_argument("--no-mask-missing", action="store_true")
    return parser.parse_args()


def run():
    """Run the experiment with user-supplied settings."""
    args = parse_args()
    main(
        test_mode=args.test_mode,
        mask_missing_during_eval=not args.no_mask_missing,
        mask_fraction=args.mask_fraction,
        use_bigrams_for_mis=not args.no_bigrams,
        use_normalization_for_mis=not args.no_normalization,
        use_unigrams=not args.no_unigrams,
        normalization_json_path=args.mapping,
        input_path=args.input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    run()

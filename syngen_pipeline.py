"""Command-line wrapper for the synthetic visibility pipeline.

Public names are re-exported for compatibility with earlier imports.
"""

import argparse
from pathlib import Path

from mf_syngen import *  # noqa: F403 - deliberate backwards-compatible API


DEFAULT_CONFIG_PATH = Path("components_multifreq.json")
DEFAULT_UVF_DIR = Path("data")
DEFAULT_OUT_DIR = Path("output") / "synthetic_vis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one synthetic uvfits file per input uvf/uvfits file using "
            "the input file's UV coverage and observing frequency."
        )
    )
    parser.add_argument(
        "-d", "--uvf-dir", type=Path, default=DEFAULT_UVF_DIR,
        help=f"Directory with input .uvf/.uvfits files. Default: {DEFAULT_UVF_DIR}",
    )
    parser.add_argument(
        "-c", "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help=f"JSON config describing components and spectral parameters. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "-o", "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Directory for synthetic outputs. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "-x", "--overwrite", action="store_true",
        help="Reuse the requested output directory and overwrite matching output files.",
    )
    parser.add_argument(
        "-i", "--if-indices", type=int, nargs="+", default=None,
        help="Override IF indices. Example: -i 0 1 2 3",
    )
    parser.add_argument(
        "--add-th-noise",
        dest="add_th_noise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ehtim thermal/noise/calibration effects. Default: enabled.",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=1,
        help="Random seed passed to ehtim when thermal noise is enabled. Default: 1",
    )
    parser.add_argument(
        "-S", "--source-check", choices=["warn", "ignore", "error"], default="warn",
        help="How to handle expected source mismatches. Default: warn",
    )
    return parser.parse_args()


def main() -> int:
    parsed_args = parse_args()
    config = load_config(parsed_args.config)  # noqa: F405
    run_pipeline(  # noqa: F405
        uvf_dir=parsed_args.uvf_dir,
        config=config,
        out_dir=parsed_args.out_dir,
        if_indices_override=parsed_args.if_indices,
        add_th_noise=parsed_args.add_th_noise,
        seed=parsed_args.seed,
        source_check=parsed_args.source_check,
        overwrite=parsed_args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

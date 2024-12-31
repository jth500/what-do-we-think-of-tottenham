"""Load images from https://www.kaggle.com/datasets/alexteboul/top-5-football-leagues-club-logos"""

import os
from argparse import ArgumentParser

from wdwtot.constants import ROOT
from wdwtot.preprocess.perturb import process_folder

if __name__ == "__main__":
    # add optional arguments for input_folder and output_folder
    parser = ArgumentParser()
    parser.add_argument(
        "--input_folder", type=str, default="data/top-5-football-leagues"
    )
    parser.add_argument("--output_folder", type=str, default="data/processed")

    args = parser.parse_args()
    input_folder = ROOT / args.input_folder
    output_folder = ROOT / args.output_folder

    os.makedirs(output_folder, exist_ok=True)
    process_folder(input_folder, output_folder)

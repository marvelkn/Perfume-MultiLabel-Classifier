"""SMILES -> Morgan 2048 + delapan deskriptor; demonstrasi satu molekul."""

import argparse
import sys

import numpy as np

from _lokasi import ROOT

sys.path.insert(0, str(ROOT))
from research.features import DESCRIPTORS8, calculate, calculate_morgan  # noqa: E402
from research.experiment import features  # noqa: E402


def periksa_fitur(smiles: str) -> None:
    morgan = calculate_morgan([smiles])
    deskriptor = calculate([smiles])
    print(f"SMILES: {smiles}")
    print(f"Morgan: {morgan.shape[1]} bit biner; bit aktif: {int(morgan.sum())}")
    for nama, nilai in zip(DESCRIPTORS8, deskriptor[0]):
        print(f"  {nama:<20} {float(nilai):.4f}")
    print(f"A/C: {features('xgb_A', morgan, deskriptor).shape[1]} fitur")
    print(f"B/D: {features('xgb_D', morgan, deskriptor).shape[1]} fitur")
    assert np.isin(morgan, [0, 1]).all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", default="COc1cc(C=O)ccc1O", help="Satu molekul valid; default vanilin")
    periksa_fitur(parser.parse_args().smiles)

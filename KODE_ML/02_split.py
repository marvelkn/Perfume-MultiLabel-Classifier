"""Lihat pembagian train/test dan satu contoh fold berkelompok tanpa training."""

import sys

import numpy as np

from _lokasi import DATA, PROTOKOL, ROOT, baca_json

sys.path.insert(0, str(ROOT))


def periksa_split() -> None:
    split = baca_json(DATA / "splits.json")
    protokol = baca_json(PROTOKOL)
    kelompok = np.asarray(split["groups"])
    train, test = np.asarray(split["train"]), np.asarray(split["test"])
    bocor = set(kelompok[train]) & set(kelompok[test])
    if bocor:
        raise AssertionError(f"Ada {len(bocor)} grup di train dan test!")

    label = split["eligible_labels"][0]
    folds = split["folds"][label]
    print(f"Molekul: {len(kelompok)}; train: {len(train)}; test: {len(test)}")
    print(f"Grup bocor train-test: {len(bocor)}")
    print(f"Contoh label: {label}; jumlah fold: {len(folds)}")
    for nomor, fold in enumerate(folds, start=1):
        a, b = np.asarray(fold["train"]), np.asarray(fold["validation"])
        if set(kelompok[a]) & set(kelompok[b]):
            raise AssertionError(f"Fold {nomor} memiliki kebocoran grup.")
        print(f"  Fold {nomor}: {len(a)} latih, {len(b)} validasi")
    print(f"Seed validasi berulang: {protokol['robustness']['seeds']}")
    print("Test tetap disimpan di luar proses tuning dan validasi.")


if __name__ == "__main__":
    periksa_split()

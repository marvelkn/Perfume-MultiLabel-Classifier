"""Prediksi satu SMILES dengan model penelitian; tidak menulis hasil."""

import argparse
import sys

from _lokasi import ROOT, hasil_penelitian

sys.path.insert(0, str(ROOT))
from alignment.predict_v7 import predict  # noqa: E402


def contoh_prediksi(smiles: str, jumlah: int) -> None:
    hasil = predict(hasil_penelitian(), smiles)
    print(f"Molekul: {hasil['smiles']} | Kandidat: {hasil['candidate']}")
    print(f"Ambang keputusan: {hasil['threshold']}")
    for baris in hasil["labels"][:jumlah]:
        print(f"{baris['label']:<20} skor={baris['score']:.4f}  prediksi={baris['predicted']}")
    print("Skor model belum dikalibrasi; ini bukan intensitas aroma.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", required=True, help="SMILES satu molekul valid")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top harus minimal 1")
    contoh_prediksi(args.smiles, args.top)

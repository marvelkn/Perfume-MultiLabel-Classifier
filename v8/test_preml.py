"""Regression tests for scientific data identity; no fitting."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from v8.prepare import prepare_checkpoint, split_diagnostic, feature_audit, verify_reference, ROOT, D8


class PreMLTests(unittest.TestCase):
    def test_missing_duplicate_and_multilabel_semantics(self):
        rows = [
            ["CCO", 1, "Sweet", "Woody", None, "mb"],
            ["CCO", 1, "Fishy", None, None, "different"],
            ["CCC", 2, "Sweet", None, None, None],
            ["CCN", 3, "Woody", "Woody", None, "mb"],
            ["CO", 4, None, None, None, "mb"],
        ]
        raw = pd.DataFrame(rows, columns=["smiles", "cid", "ifra1", "ifra2", "ifra3", "molblock"])
        frame, truth, labels, _, support, _, rejected = prepare_checkpoint(raw)
        self.assertEqual(frame.source_row.tolist(), [0, 3])
        self.assertEqual(labels, ["SWEET", "WOODY"])
        np.testing.assert_array_equal(truth, [[1, 1], [0, 1]])
        self.assertEqual(support.set_index("label").loc["WOODY", "slot_occurrence_count"], 3)
        self.assertEqual(support.set_index("label").loc["WOODY", "record_positive_count"], 2)
        self.assertEqual(len(rejected), 3)

    def test_split_bug_reports_cohort_not_false_xy_mismatch(self):
        labels = ["A"] * 5 + ["RARE"] + ["B"] * 5 + ["C"] * 5
        n = len(labels)
        frame = pd.DataFrame({"source_row": np.arange(100, 100+n), "stratify_label": labels})
        y = np.zeros((n, 4), np.uint8)
        for i, name in enumerate(labels):
            y[i, ["A", "B", "C", "RARE"].index(name)] = 1
        matrices = {"reference_morgan_1024": np.arange(n).reshape(n, 1),
                    "morgan_2048": np.arange(n).reshape(n, 1),
                    "labels": np.array(["A", "B", "C", "RARE"])}
        table, audit, _ = split_diagnostic(frame, y, matrices, list(range(n)), list(range(n)))
        self.assertGreater(audit["mismatched_source_row_positions"], 0)
        self.assertEqual(audit["actual_cohort_rows_outside_intended"], 1)
        self.assertFalse(audit["index_bug_corrected"])
        np.testing.assert_array_equal(table.actual_feature_source_row, table.actual_label_source_row)

    def test_modified_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "author_code.ipynb").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_reference(temp)

    def test_v7_feature_recipe_and_reference_encoding(self):
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdFingerprintGenerator
        rows = [
            ["CCO", 1, "SWEET", None, None, Chem.MolToMolBlock(Chem.MolFromSmiles("CCO"))],
            ["CC(=O)O", 2, "SOUR", "SWEET", None, Chem.MolToMolBlock(Chem.MolFromSmiles("CC(=O)O"))],
            ["[Na+].[Cl-]", 3, "ODORLESS", None, None, Chem.MolToMolBlock(Chem.MolFromSmiles("[Na+].[Cl-]"))],
        ]
        raw = pd.DataFrame(rows, columns=["smiles", "cid", "ifra1", "ifra2", "ifra3", "molblock"])
        frame, y, labels, *_ = prepare_checkpoint(raw)
        matrices, audit, *_ = feature_audit(frame, y, labels, ROOT/"data/reference/suh2025")
        self.assertEqual(matrices["X_v8_2056"].shape, (3, 2056))
        self.assertEqual(audit["multicomponent_records"], 1)
        self.assertEqual(len(frame), 3)
        self.assertFalse(any(a.dtype.hasobject for a in matrices.values()))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "features.npz"
            np.savez_compressed(path, **matrices)
            with np.load(path, allow_pickle=False) as saved:
                for key, value in matrices.items():
                    np.testing.assert_array_equal(saved[key], value)
        mol = Chem.MolFromSmiles("CCO")
        expected = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048, includeChirality=False).GetFingerprintAsNumPy(mol)
        np.testing.assert_array_equal(matrices["morgan_2048"][0], expected)
        np.testing.assert_array_equal(matrices["descriptors_8"][0],
                                      np.array([getattr(Descriptors, x)(mol) for x in D8], dtype=np.float32))
        self.assertTrue(audit["label_encoding_matches_author_helpers"])


if __name__ == "__main__":
    unittest.main()

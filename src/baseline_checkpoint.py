"""Versioned baseline prediction checkpoints; incomplete labels are recomputed."""
import json
from pathlib import Path

import numpy as np

from .artifacts import digest, write_json


class BaselineCheckpoint:
    def __init__(self, run, algorithm, signature, strategies, folds, labels):
        self.run = Path(run)
        self.path = self.run / f"{algorithm}_baseline_progress.json"
        self.directory = self.run / f"{algorithm}_baseline_checkpoints"
        self.expected = {f"{strategy}/{fold}/{label}": len(split["score"])
                         for strategy in strategies for fold, split in enumerate(folds)
                         for label in range(labels)}
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
            if self.state.get("version") != 1 or self.state.get("signature") != signature:
                raise ValueError("Baseline progress protocol/code differs or is unversioned; use a new run")
            entries = self.state.get("entries")
            if not isinstance(entries, dict) or not set(entries) <= self.expected.keys():
                raise ValueError("Invalid baseline checkpoint keys")
        else:
            if self.directory.exists():
                raise ValueError("Unrecognized baseline checkpoint directory")
            self.state = {"version": 1, "signature": signature, "entries": {}, "results": {}}
            write_json(self.path, self.state)

    def _path(self, key):
        if key not in self.expected:
            raise ValueError("Unexpected baseline checkpoint key")
        return self.directory / (key + ".npz")

    def load(self, key):
        item = self.state["entries"].get(key)
        if item is None:
            return None
        path = self._path(key)
        if not isinstance(item, dict) or not path.is_file() or digest(path) != item.get("sha256"):
            raise ValueError("Baseline prediction checkpoint integrity failure")
        with np.load(path, allow_pickle=False) as payload:
            predictions = payload["probabilities"]
        rounds = item.get("rounds")
        if predictions.shape != (self.expected[key],) or not np.isfinite(predictions).all() or not ((predictions >= 0) & (predictions <= 1)).all():
            raise ValueError("Invalid baseline prediction checkpoint values")
        if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
            raise ValueError("Invalid baseline iteration checkpoint")
        return predictions, rounds

    def save(self, key, predictions, rounds):
        path = self._path(key)
        predictions = np.asarray(predictions)
        if predictions.shape != (self.expected[key],) or not np.isfinite(predictions).all() or not ((predictions >= 0) & (predictions <= 1)).all():
            raise ValueError("Invalid baseline predictions; checkpoint not saved")
        if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
            raise ValueError("Invalid baseline iteration count")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, probabilities=predictions)
        temporary.replace(path)
        self.state["entries"][key] = {"sha256": digest(path), "rounds": rounds}
        write_json(self.path, self.state)

    def record_results(self, results):
        # Scores are recomputed from verified predictions when resuming.
        self.state["results"] = results
        write_json(self.path, self.state)

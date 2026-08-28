"""Harmonize GoodScents free-text odor descriptors onto the Leffingwell taxonomy.

The Leffingwell archive ships a clean one-hot taxonomy (~113 odor classes). GoodScents
ships free text ("sweet;vanilla;cherry maraschino cherry;..."). This module maps the
free text onto the taxonomy via: exact match -> curated synonym -> unmapped (reported).

A mapped synonym only takes effect if its TARGET exists in the live taxonomy, so wrong
guesses in SYNONYMS are harmless no-ops. Extend SYNONYMS from the generated
data/interim/unmapped_descriptors.csv report (see plan, Action-Plan Task 3).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Optional

import pandas as pd

# Starter synonym map (lowercased GoodScents form -> Leffingwell class).
# Validated against the live taxonomy at runtime; extend after reviewing the unmapped report.
SYNONYMS = {
    "fruit": "fruity", "fruits": "fruity",
    "flower": "floral", "flowers": "floral", "floral notes": "floral",
    "wood": "woody", "woods": "woody",
    "herb": "herbal", "herbaceous": "herbal",
    "green notes": "green", "citrus notes": "citrus",
    "spice": "spicy", "spicy notes": "spicy",
    "sweet notes": "sweet",
    "creamy": "cream", "buttery": "butter",
    "nutty": "nut", "smoky": "smoke", "smokey": "smoke",
    "earthy": "earth", "minty": "mint",
    "balsam": "balsamic", "powder": "powdery",
    "oily": "oil", "meaty": "meat", "winey": "wine", "wine like": "wine",
}

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def normalize(token: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    t = str(token).strip().lower()
    t = _NON_ALNUM.sub(" ", t)
    return _WS.sub(" ", t).strip()


def split_descriptors(cell: str) -> list[str]:
    """Split a semicolon-separated GoodScents Descriptors cell into normalized tokens."""
    if not isinstance(cell, str):
        return []
    return [n for n in (normalize(p) for p in cell.split(";")) if n]


class LabelHarmonizer:
    """Maps free-text descriptors onto a fixed taxonomy and tracks what didn't map."""

    def __init__(self, taxonomy: Iterable[str]):
        self.taxonomy = [normalize(t) for t in taxonomy]
        self._tax = set(self.taxonomy)
        self._syn = {normalize(k): normalize(v) for k, v in SYNONYMS.items()}
        self.unmapped: Counter = Counter()

    def map_descriptor(self, descriptor: str) -> Optional[str]:
        d = normalize(descriptor)
        if not d:
            return None
        if d in self._tax:
            return d
        syn = self._syn.get(d)
        if syn and syn in self._tax:
            return syn
        self.unmapped[d] += 1
        return None

    def map_row(self, cell: str) -> set:
        """Map one GoodScents Descriptors cell to a set of taxonomy labels."""
        out = set()
        for desc in split_descriptors(cell):
            m = self.map_descriptor(desc)
            if m:
                out.add(m)
        return out

    def unmapped_report(self) -> pd.DataFrame:
        """Descriptors that didn't map, ranked by frequency (hand-curate the top ones)."""
        return pd.DataFrame(
            sorted(self.unmapped.items(), key=lambda kv: kv[1], reverse=True),
            columns=["descriptor", "count"],
        )

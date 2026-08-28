"""Perfume aroma-profile prediction — shared data layer (Marvel Kevin Nathanael's thesis).

Pipeline: SMILES (GoodScents + Leffingwell) -> canonicalize/dedupe -> harmonize labels
to the Leffingwell taxonomy -> Morgan fingerprint (r=2, 2048-bit) -> iterative-stratified
80/20 split. Modeling (XGBoost/LightGBM, Binary Relevance) starts downstream of this layer.
"""

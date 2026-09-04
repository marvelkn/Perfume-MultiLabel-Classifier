"""Essenza molecular ML pipeline. Limit native parallelism before importing NumPy."""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = os.environ.get("ESSENZA_THREADS", "2")

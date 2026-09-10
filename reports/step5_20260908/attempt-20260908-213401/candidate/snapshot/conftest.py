"""Ensure the project root is importable as `src` during tests."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

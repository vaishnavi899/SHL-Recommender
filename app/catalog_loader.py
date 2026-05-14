"""
catalog_loader.py — Loads the SHL product catalog JSON from disk.

The catalog JSON is expected to be a list of dicts, each with at minimum:
  name, description, link, keys, job_levels
"""

import json
import os


def load_catalog(path: str = "data/shl_product_catalog.json") -> list:
    """
    Load catalog from `path`. Raises a clear error if the file is missing
    so startup fails fast rather than silently serving an empty catalog.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SHL catalog not found at '{path}'. "
            "Run the catalog scraper first, or check the data/ directory."
        )

    with open(path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    if not isinstance(catalog, list) or len(catalog) == 0:
        raise ValueError(
            f"Catalog at '{path}' is empty or not a list. "
            "Verify the scraper output."
        )

    return catalog
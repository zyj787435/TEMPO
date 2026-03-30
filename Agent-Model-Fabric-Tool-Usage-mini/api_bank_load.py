"""Load API-Bank CSV and compute capability vectors for each API."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from text_handling import CAPABILITY_ORDER, compute_keyword_vector, normalize_text

DEFAULT_CSV = Path(__file__).resolve().parent / "all_apis.csv"


@dataclass
class APIDoc:
	id: str
	name: str
	description: str
	a_api: np.ndarray


def _compose_description(row: dict) -> str:
	parts: List[str] = []
	for key in ("应用场景", "API名称", "api_info"):
		val = row.get(key)
		if val:
			parts.append(str(val))
	return " | ".join(parts)


def _tokens_from_text(text: str) -> List[str]:
	return normalize_text(text).split()


def compute_api_vector(text: str) -> np.ndarray:
	tokens = _tokens_from_text(text)
	return compute_keyword_vector(tokens)


def load_api_bank(csv_path: Optional[str] = None) -> List[APIDoc]:
	path = Path(csv_path) if csv_path else DEFAULT_CSV
	docs: List[APIDoc] = []

	with path.open(newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			api_id = row.get("id") or ""
			name = row.get("API名称") or row.get("name") or ""
			desc = _compose_description(row)
			a_api = compute_api_vector(desc)
			docs.append(APIDoc(id=str(api_id), name=name, description=desc, a_api=a_api))

	return docs


__all__ = ["APIDoc", "load_api_bank", "compute_api_vector", "CAPABILITY_ORDER"]

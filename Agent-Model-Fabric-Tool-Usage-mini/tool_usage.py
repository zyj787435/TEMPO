"""Tool selection: score APIs against task vectors."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from api_bank_load import APIDoc, load_api_bank


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
	if u.size == 0 or v.size == 0:
		return 0.0
	denom = np.linalg.norm(u) * np.linalg.norm(v)
	if denom == 0:
		return 0.0
	return float(np.dot(u, v) / denom)


def score_apis(a_t: np.ndarray, z_sem: np.ndarray | None = None, top_k: int = 5) -> List[Tuple[APIDoc, float]]:
	"""Compute cosine scores between task capability vector and APIs.

	z_sem is accepted for future extensions; current scoring uses a_T vs a_API.
	"""

	apis = load_api_bank()
	scored = [(api, _cosine(a_t, api.a_api)) for api in apis]
	scored.sort(key=lambda x: x[1], reverse=True)
	return scored[:top_k]


__all__ = ["score_apis"]

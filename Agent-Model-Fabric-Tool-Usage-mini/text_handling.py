"""Text normalization and task representation utilities.

This module produces:
- normalized text `~T`
- semantic embedding `z_sem`
- functional requirement vector `a_T`
- fused representation `z_T = concat(z_sem, lambda * a_T)`
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from keywords_conf import KEYWORDS


# Order of capabilities for a_T / a_API
CAPABILITY_ORDER: Tuple[str, ...] = ("reason", "code", "math", "tool", "domain")


@dataclass
class TaskVectors:
	normalized_text: str
	a_t: np.ndarray
	z_sem: np.ndarray
	z_t: np.ndarray


def normalize_text(text: str) -> str:
	"""Lowercase and strip non-alphanumeric chars while keeping spaces."""
	lowered = text.lower()
	cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
	normalized = re.sub(r"\s+", " ", cleaned).strip()
	return normalized


def _tokenize(text: str) -> List[str]:
	return normalize_text(text).split()


def _sigmoid(x: np.ndarray) -> np.ndarray:
	return 1 / (1 + np.exp(-x))


def compute_keyword_vector(tokens: Iterable[str]) -> np.ndarray:
	"""Count keyword hits per capability and squash with sigmoid."""
	counts = []
	token_set = list(tokens)
	for key in CAPABILITY_ORDER:
		vocab = KEYWORDS.get(key, [])
		hits = sum(1 for t in token_set if t in vocab)
		counts.append(hits)
	return _sigmoid(np.array(counts, dtype=float))


def _load_sentence_model():
	try:
		from sentence_transformers import SentenceTransformer

		return SentenceTransformer("all-MiniLM-L6-v2")
	except Exception:
		return None


_SBERT_MODEL = _load_sentence_model()


def compute_semantic_embedding(text: str) -> np.ndarray:
	"""Return semantic embedding; fallback to simple hash vector if SBERT unavailable."""

	if _SBERT_MODEL is not None:
		try:
			vec = _SBERT_MODEL.encode(text, normalize_embeddings=True)
			return np.asarray(vec, dtype=float)
		except Exception:
			pass

	# Fallback: hash-based 64-dim bag-of-words style vector
	tokens = _tokenize(text)
	dim = 64
	vec = np.zeros(dim, dtype=float)
	for tok in tokens:
		h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
		idx = h % dim
		vec[idx] += 1.0
	if np.linalg.norm(vec) > 0:
		vec = vec / np.linalg.norm(vec)
	return vec


def fuse_vectors(z_sem: np.ndarray, a_t: np.ndarray, lam: float = 1.0) -> np.ndarray:
	return np.concatenate([z_sem, lam * a_t])


def process_task(text: str, lam: float = 1.0) -> TaskVectors:
	normalized = normalize_text(text)
	tokens = normalized.split()
	a_t = compute_keyword_vector(tokens)
	z_sem = compute_semantic_embedding(normalized)
	z_t = fuse_vectors(z_sem, a_t, lam)
	return TaskVectors(normalized_text=normalized, a_t=a_t, z_sem=z_sem, z_t=z_t)

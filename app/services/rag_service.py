from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app.config.settings import get_settings

# Tentative de chargement sentence-transformers (NLP semantique)
# Fallback automatique TF-IDF si bibliotheque absente
try:
	from sentence_transformers import SentenceTransformer  # type: ignore
	_HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
	_HAVE_SENTENCE_TRANSFORMERS = False
	from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class RagDoc:
	id: str
	text: str
	metadata: dict[str, Any]


class RagService:
	"""Service RAG avec embeddings semantiques (sentence-transformers).
	Fallback automatique sur TF-IDF si la bibliotheque n'est pas installee.
	"""

	def __init__(self) -> None:
		self.settings = get_settings()
		self._encoder: Any | None = None
		self._vectorizer: Any | None = None
		self._matrix: Any | None = None
		self._docs: list[RagDoc] = []
		self._dataset_path: Path | None = self.settings.dataset_path
		self._use_semantic = False

		if _HAVE_SENTENCE_TRANSFORMERS:
			model_name = self.settings.embedding_model
			print(f"[RAG] Chargement du modele NLP semantique : {model_name}")
			try:
				self._encoder = SentenceTransformer(model_name)
				self._use_semantic = True
				print("[RAG] Modele semantique pret (sentence-transformers)")
			except Exception as e:
				print(f"[RAG] Erreur chargement modele : {e} => fallback TF-IDF")
		else:
			print("[RAG] sentence-transformers non installe => TF-IDF active")

		if self._dataset_path and self._dataset_path.exists():
			print(f"[RAG] Chargement du dataset : {self._dataset_path}")
			try:
				self.build_index(
					dataset_path=self._dataset_path,
					text_columns=self.settings.dataset_text_columns.split(',') if self.settings.dataset_text_columns else None,
					max_rows=self.settings.rag_max_rows,
				)
				print(f"[RAG] Index pret ! {len(self._docs)} documents indexes.")
			except Exception as e:
				print(f"[RAG] Erreur au chargement : {e}")
		else:
			print("[RAG] Aucun dataset configure dans .env (DATASET_PATH)")

	@property
	def is_ready(self) -> bool:
		return bool(self._docs) and self._matrix is not None

	def build_index(
		self,
		dataset_path: Path,
		text_columns: list[str] | None = None,
		max_rows: int = 5000,
	) -> None:
		dataset_path = Path(dataset_path)
		if not dataset_path.exists():
			raise FileNotFoundError(f"Dataset not found: {dataset_path}")

		df = pd.read_csv(dataset_path)
		if max_rows and len(df) > max_rows:
			df = df.head(max_rows)

		priority_cols = [
			'Department',
			'JobRole',
			'MainDomain',
			'Skills',
			'Seniority',
			'Certifications',
			'ProjectHistory',
			'AttritionReason',
			'AbsenceDaysLast12Months',
			'MainAbsenceType',
		]
		existing_cols = [c for c in priority_cols if c in df.columns]
		if not existing_cols:
			print("[RAG WARNING] Aucune colonne prioritaire => fallback toutes colonnes texte")
			existing_cols = [c for c in df.columns if df[c].dtype == 'object']

		docs: list[RagDoc] = []
		texts: list[str] = []
		for idx, row in df.iterrows():
			parts: list[str] = []
			for c in existing_cols:
				v = row.get(c)
				if pd.isna(v):
					continue
				s = str(v).strip()
				if s:
					parts.append(f"{c}: {s}")
			text = "\n".join(parts).strip()
			if not text:
				continue
			docs.append(RagDoc(id=str(idx), text=text, metadata={"row_index": int(idx)}))
			texts.append(text)

		# Construction matrice de similarite
		if self._use_semantic and self._encoder is not None:
			print(f"[RAG] Calcul des embeddings semantiques pour {len(texts)} documents...")
			embeddings = self._encoder.encode(
				texts,
				show_progress_bar=True,
				batch_size=64,
				convert_to_numpy=True,
			)
			self._matrix = embeddings
			print(f"[RAG] Embeddings calcules : shape={self._matrix.shape}")
		else:
			from sklearn.feature_extraction.text import TfidfVectorizer
			vectorizer = TfidfVectorizer(
				lowercase=True,
				strip_accents="unicode",
				max_features=50000,
				ngram_range=(1, 2),
			)
			self._matrix = vectorizer.fit_transform(texts)
			self._vectorizer = vectorizer

		self._docs = docs
		self._dataset_path = dataset_path
		mode = "semantique (NLP)" if self._use_semantic else "TF-IDF (fallback)"
		print(f"[RAG] Index reconstruit en mode {mode} | colonnes : {existing_cols}")

	def retrieve(self, query: str, top_k: int = 5) -> list[tuple[RagDoc, float]]:
		if not self.is_ready:
			return []
		query = (query or '').strip()
		if not query:
			return []

		if self._use_semantic and self._encoder is not None:
			query_emb = self._encoder.encode([query], convert_to_numpy=True)
			sims = cosine_similarity(query_emb, self._matrix).ravel()
		else:
			assert self._vectorizer is not None
			q = self._vectorizer.transform([query])
			sims = cosine_similarity(q, self._matrix).ravel()

		if sims.size == 0:
			return []

		k = max(1, min(int(top_k or 5), len(self._docs)))
		best_idx = sims.argsort()[-k:][::-1]
		out: list[tuple[RagDoc, float]] = []
		for i in best_idx:
			score = float(sims[i])
			if score <= 0.0:
				continue
			out.append((self._docs[int(i)], score))
		return out

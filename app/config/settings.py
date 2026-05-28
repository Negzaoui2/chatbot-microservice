from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
	model_config = SettingsConfigDict(
		env_file=".env",
		env_file_encoding="utf-8",
		extra="ignore",
	)

	app_name: str = Field(default="chatbot-microservice")
	environment: str = Field(default="dev")
	log_level: str = Field(default="INFO")

	# Dataset / RAG
	dataset_path: Path | None = Field(
		default=None,
		description="Path to a local CSV file (e.g., Kaggle HR dataset exported as CSV).",
	)
	dataset_text_columns: str = Field(
		default="",
		description="Comma-separated list of columns used to build searchable text.",
	)
	rag_top_k: int = Field(default=5)
	rag_max_rows: int = Field(
		default=5000,
		description="Safety cap to avoid indexing extremely large CSVs by accident.",
	)

	# Optional LLM (Groq)
	groq_api_key: str | None = Field(default=None)
	groq_model: str = Field(default="llama-3.3-70b-versatile")
	llm_timeout_seconds: float = Field(default=30.0)

	# CORS — origines autorisées (séparées par virgule dans .env)
	cors_origins: str = Field(
		default="http://localhost:4200,http://localhost:4201",
		description="Comma-separated list of allowed CORS origins (Angular dev server).",
	)

	# Embeddings sémantiques (sentence-transformers)
	embedding_model: str = Field(
		default="paraphrase-multilingual-MiniLM-L12-v2",
		description="Modèle sentence-transformers pour le RAG sémantique.",
	)

	# Historique conversationnel
	max_history_turns: int = Field(
		default=8,
		description="Nombre maximum de tours (user + assistant) gardés en mémoire par session.",
	)


@lru_cache
def get_settings() -> Settings:
	return Settings()

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
	status: str = "ok"


class RetrievedItem(BaseModel):
	id: str
	score: float
	text: str
	metadata: dict[str, Any] = Field(default_factory=dict)


class HistoryMessage(BaseModel):
	"""Un tour de conversation (user ou assistant)."""
	role: Literal["user", "assistant"]
	content: str


class ChatRequest(BaseModel):
	message: str = Field(min_length=1)
	session_id: str | None = None
	# Historique optionnel envoyé par le client (si pas de session_id)
	history: list[HistoryMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
	answer: str
	intent: str = "question"
	retrieved: list[RetrievedItem] = Field(default_factory=list)
	used_llm: bool = False
	session_id: str | None = None
	download_url: str | None = None

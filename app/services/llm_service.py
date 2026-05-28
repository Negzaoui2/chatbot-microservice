from __future__ import annotations

from dataclasses import dataclass

import httpx
from groq import Groq


@dataclass(frozen=True)
class LlmResult:
	text: str
	used_llm: bool


class LlmService:
	def __init__(self, api_key: str | None, model: str, timeout_seconds: float = 30.0) -> None:
		self._api_key = api_key
		self._model = model
		self._timeout = timeout_seconds
		self._client = None
		if self.enabled:
			self._client = Groq(
				api_key=self._api_key,
				http_client=httpx.Client(verify=False),
			)
			print(f"[LLM INIT] Groq client créé avec clé commençant par : {self._api_key[:10]}...")
		else:
			print("[LLM INIT] Pas de clé API → LLM désactivé")
	@property
	def enabled(self) -> bool:
		return bool(self._api_key)

	def generate(self, system_prompt: str, user_prompt: str) -> LlmResult:
		print("[LLM DEBUG] === Début generate() ===")
		print(f"[LLM DEBUG] Clé existe ? {bool(self._api_key)}")
		print(f"[LLM DEBUG] Client existe ? {self._client is not None}")
		print(f"[LLM DEBUG] Modèle demandé : {self._model}")

		if not self.enabled:
			print("[LLM] LLM désactivé (pas de clé)")
			return LlmResult(text="", used_llm=False)

		if self._client is None:
			print("[LLM CRITICAL] Client Groq est None malgré enabled=True !")
			return LlmResult(text="", used_llm=False)

		try:
			print("[LLM] Envoi de la requête à Groq...")
			print(f"[LLM] Prompt user (début) : {user_prompt[:150]}...")
			print(f"[LLM] Prompt système (début) : {system_prompt[:150]}...")

			resp = self._client.chat.completions.create(
				model=self._model,
				messages=[
					{"role": "system", "content": system_prompt},
					{"role": "user", "content": user_prompt},
				],
				timeout=self._timeout,
				temperature=0.7,
				max_tokens=500,
			)

			print("[LLM] Réponse reçue de Groq !")
			content = (resp.choices[0].message.content or "").strip()
			print(f"[LLM] Contenu reçu (début) : {content[:200]}...")
			return LlmResult(text=content, used_llm=True)

		except Exception as e:
			import traceback
			print("[LLM CRITICAL ERROR] Groq a échoué pendant l'appel !")
			print(f"[LLM ERROR] Message exact : {str(e)}")
			print(f"[LLM ERROR] Type d'exception : {type(e).__name__}")
			traceback.print_exc()  # ← affiche toute la pile d'erreur
			return LlmResult(text="", used_llm=False)
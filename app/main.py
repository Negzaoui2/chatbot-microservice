from __future__ import annotations

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config.settings import get_settings
from app.models.schemas import ChatRequest, ChatResponse, HistoryMessage, HealthResponse, RetrievedItem
from app.services.analytics_service import AnalyticsService
from app.services.export_service import ExportService
from app.services.intent_detector import IntentDetector
from app.services.llm_service import LlmService
from app.services.planning_service import PlanningService
from app.services.rag_service import RagService
from app.utils.prompts import build_system_prompt, build_user_prompt


settings = get_settings()
app = FastAPI(title=settings.app_name)

# ── CORS — requis pour le frontend Angular ────────────────────────────────────
allowed_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
	CORSMiddleware,
	allow_origins=allowed_origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# ── Historique conversationnel par session (mémoire in-process) ──────────────
# Clé : session_id | Valeur : liste de HistoryMessage
_session_store: dict[str, list[HistoryMessage]] = {}


def _get_history(session_id: str | None) -> list[HistoryMessage]:
	if not session_id:
		return []
	return _session_store.setdefault(session_id, [])


def _save_to_history(
	session_id: str | None,
	user_msg: str,
	assistant_msg: str,
) -> None:
	if not session_id:
		return
	history = _session_store.setdefault(session_id, [])
	history.append(HistoryMessage(role="user", content=user_msg))
	history.append(HistoryMessage(role="assistant", content=assistant_msg))
	# Garder seulement les N derniers tours
	max_turns = settings.max_history_turns * 2  # *2 car user+assistant
	if len(history) > max_turns:
		_session_store[session_id] = history[-max_turns:]

# ── Services ──────────────────────────────────────────────────────────
rag_service = RagService()
intent_detector = IntentDetector()
analytics_service = AnalyticsService()
export_service = ExportService()
planning_service = PlanningService()
llm_service = LlmService(
	api_key=settings.groq_api_key,
	model=settings.groq_model,
	timeout_seconds=settings.llm_timeout_seconds,
)


@app.on_event("startup")
def _startup() -> None:
	print("[STARTUP] Démarrage du microservice...")
	if settings.dataset_path:
		text_cols = [c.strip() for c in settings.dataset_text_columns.split(",") if c.strip()]
		rag_service.build_index(
			dataset_path=settings.dataset_path,
			text_columns=text_cols,
			max_rows=settings.rag_max_rows,
		)
	print(f"[STARTUP] RAG ready: {rag_service.is_ready}")
	print(f"[STARTUP] Analytics ready: {analytics_service.is_ready}")
	print(f"[STARTUP] Planning ready: {planning_service.is_ready}")
	print(f"[STARTUP] LLM enabled: {llm_service.enabled}")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
	return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
	intent = intent_detector.detect(payload.message)
	print(f"[CHAT] Intent détecté : {intent}")

	# Récupère l'historique : priorité session serveur, fallback historique client
	history = _get_history(payload.session_id) or payload.history

	if intent == "export":
		response = _handle_export(payload.message)
	elif intent == "report":
		response = _handle_report(payload.message, intent, history)
	elif intent in ("suggestion", "optimization"):
		response = _handle_planning(payload.message, intent, history)
	else:
		response = _handle_rag_question(payload.message, intent, history)

	# Sauvegarde dans l'historique de session
	_save_to_history(payload.session_id, payload.message, response.answer)
	response.session_id = payload.session_id
	return response


def _handle_export(message: str) -> ChatResponse:
	"""Détecte le format et le type de rapport, renvoie un lien de téléchargement."""
	text = message.lower()

	# Déterminer le format
	if "excel" in text or "xlsx" in text:
		fmt = "excel"
	else:
		fmt = "pdf"

	# Déterminer le type de rapport
	report_type = "global"
	if any(k in text for k in ["attrition", "départ", "turnover"]):
		report_type = "attrition"
	elif any(k in text for k in ["effectif", "headcount", "rôle", "role"]):
		report_type = "headcount"
	elif any(k in text for k in ["absence", "congé", "maladie"]):
		report_type = "absence"
	elif any(k in text for k in ["salaire", "rémunération", "paie"]):
		report_type = "salary"
	elif any(k in text for k in ["contrat", "cdi", "cdd"]):
		report_type = "contract"

	download_url = f"/export/{fmt}?report_type={report_type}"

	answer = (
		f"Votre rapport **{report_type}** en **{fmt.upper()}** est prêt !\n\n"
		f"Cliquez sur le bouton ci-dessous pour télécharger le fichier."
	)
	return ChatResponse(answer=answer, intent="export", used_llm=False, download_url=download_url)


def _handle_report(message: str, intent: str, history: list[HistoryMessage]) -> ChatResponse:
	raw_report = analytics_service.run_report(message)
	if llm_service.enabled:
		system_prompt = build_system_prompt(intent)
		user_prompt = build_user_prompt(message, raw_report, intent, history)
		result = llm_service.generate(system_prompt=system_prompt, user_prompt=user_prompt)
		if result.text:
			return ChatResponse(answer=result.text, intent=intent, used_llm=True)
	return ChatResponse(answer=raw_report, intent=intent, used_llm=False)


def _handle_planning(message: str, intent: str, history: list[HistoryMessage]) -> ChatResponse:
	raw_data = planning_service.suggest(message)
	if llm_service.enabled:
		system_prompt = build_system_prompt(intent)
		user_prompt = build_user_prompt(message, raw_data, intent, history)
		result = llm_service.generate(system_prompt=system_prompt, user_prompt=user_prompt)
		if result.text:
			return ChatResponse(answer=result.text, intent=intent, used_llm=True)
	return ChatResponse(answer=raw_data, intent=intent, used_llm=False)


def _handle_rag_question(message: str, intent: str, history: list[HistoryMessage]) -> ChatResponse:
	retrieved = rag_service.retrieve(message, top_k=settings.rag_top_k)
	retrieved_items = [
		RetrievedItem(
			id=doc.id,
			score=score,
			text=doc.text,
			metadata={**doc.metadata, "intent": intent},
		)
		for doc, score in retrieved
	]
	context = "\n\n---\n\n".join([
		f"Employé ID [{it.id}] :\n{it.text}"
		for it in retrieved_items
	])
	if llm_service.enabled:
		system_prompt = build_system_prompt(intent)
		user_prompt = build_user_prompt(message, context, intent, history)
		result = llm_service.generate(system_prompt=system_prompt, user_prompt=user_prompt)
		if result.text:
			return ChatResponse(
				answer=result.text, intent=intent, retrieved=retrieved_items, used_llm=True,
			)

	if not retrieved_items:
		return ChatResponse(
			answer="Je n'ai pas trouvé d'info pertinente. Peux-tu préciser ta question ?",
			intent=intent, retrieved=[], used_llm=False,
		)

	answer = "Voici ce que j'ai trouvé dans le dataset :\n\n" + context
	return ChatResponse(answer=answer, intent=intent, retrieved=retrieved_items, used_llm=False)


# ── Export PDF / Excel ─────────────────────────────────────────────────────────

@app.get("/export/excel")
def export_excel(
	report_type: str = Query(default="global", description="Type de rapport: global, attrition, headcount, absence, salary, contract"),
):
	"""Télécharger un rapport Excel."""
	if not export_service.is_ready:
		return {"error": "Dataset non disponible."}
	buffer = export_service.export_excel(report_type)
	filename = f"rapport_{report_type}.xlsx"
	return StreamingResponse(
		buffer,
		media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
		headers={"Content-Disposition": f"attachment; filename={filename}"},
	)


@app.get("/export/pdf")
def export_pdf(
	report_type: str = Query(default="global", description="Type de rapport: global, attrition, headcount, absence, salary, contract"),
):
	"""Télécharger un rapport PDF."""
	if not export_service.is_ready:
		return {"error": "Dataset non disponible."}
	buffer = export_service.export_pdf(report_type)
	filename = f"rapport_{report_type}.pdf"
	return StreamingResponse(
		buffer,
		media_type="application/pdf",
		headers={"Content-Disposition": f"attachment; filename={filename}"},
	)

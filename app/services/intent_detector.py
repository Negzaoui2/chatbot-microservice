from __future__ import annotations


# Intents supportés :
#   export          → Demande d'export fichier PDF/Excel (Export)
#   question        → Question RH générale (RAG + LLM)
#   report          → Demande de rapport / statistiques (Analytics)
#   suggestion      → Qui est disponible ? Affecter quelqu'un (Planning)
#   optimization    → Optimiser la charge, redistribuer (Planning)
#   leave_absence   → Questions sur congés / absences (RAG + LLM)
#   payroll_contract→ Salaire, contrat, fiche de paie (RAG + LLM)

class IntentDetector:
	"""Détecteur d'intent rule-based enrichi pour le chatbot RH."""

	EXPORT_KEYWORDS = [
		"pdf", "excel", "xlsx", "télécharger", "telecharger",
		"exporter", "export", "fichier", "download",
		"générer un rapport", "generer un rapport",
	]

	REPORT_KEYWORDS = [
		"rapport", "report", "statistique", "stats", "taux", "combien",
		"répartition", "distribution", "bilan", "synthèse", "tableau",
		"effectif", "headcount", "kpi", "indicateur",
		"département", "departement",
	]

	SUGGESTION_KEYWORDS = [
		"disponible", "dispo", "libre", "affecter", "qui peut",
		"mission", "assigner", "trouver quelqu'un",
		"recommande", "suggère", "propose",
	]

	# "projet" seul = question générale ; combiné avec un verbe d'affectation = suggestion
	SUGGESTION_COMPOUND = [
		("projet", "affecter"), ("projet", "assigner"), ("projet", "staffing"),
		("projet", "qui peut"), ("projet", "disponible"), ("projet", "dispo"),
	]

	OPTIMIZATION_KEYWORDS = [
		"optimise", "optimiser", "charge", "surcharge", "redistribu",
		"répartir", "équilibrer", "planifier", "planning",
		"staffing", "stuffing", "shift", "horaire",
	]

	LEAVE_KEYWORDS = [
		"congé", "vacances", "absence", "maladie", "arrêt",
		"rtt", "repos", "indisponible",
	]

	PAYROLL_KEYWORDS = [
		"contrat", "salaire", "paie", "fiche", "rémunération",
		"cdi", "cdd", "revenu", "prime", "augmentation",
	]

	def detect(self, message: str) -> str:
		text = (message or "").lower()

		# Export PDF/Excel détecté en priorité
		if any(k in text for k in self.EXPORT_KEYWORDS):
			return "export"
		if any(k in text for k in self.REPORT_KEYWORDS):
			return "report"
		# Suggestion : mots simples OU combinaisons composées
		if any(k in text for k in self.SUGGESTION_KEYWORDS):
			return "suggestion"
		if any(a in text and b in text for a, b in self.SUGGESTION_COMPOUND):
			return "suggestion"
		if any(k in text for k in self.OPTIMIZATION_KEYWORDS):
			return "optimization"
		if any(k in text for k in self.LEAVE_KEYWORDS):
			return "leave_absence"
		if any(k in text for k in self.PAYROLL_KEYWORDS):
			return "payroll_contract"
		return "question"

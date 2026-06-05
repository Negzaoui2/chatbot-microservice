from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.config.settings import get_settings


class AnalyticsService:
	"""Exécute des requêtes analytiques (rapports) sur le dataset HR."""

	def __init__(self) -> None:
		self._df: pd.DataFrame | None = None
		settings = get_settings()
		if settings.dataset_path and Path(settings.dataset_path).exists():
			self._df = pd.read_csv(settings.dataset_path)
			print(f"[ANALYTICS] Dataset chargé : {self._df.shape}")

	@property
	def is_ready(self) -> bool:
		return self._df is not None and not self._df.empty

	def attrition_by_department(self) -> str:
		if not self.is_ready:
			return "Dataset non chargé."
		df = self._df
		total = df.groupby("Department")["Attrition"].count()
		left = df[df["Attrition"] == "Yes"].groupby("Department")["Attrition"].count()
		rate = ((left / total) * 100).fillna(0).round(1)
		lines = ["**Taux d'attrition par département :**"]
		for dept in rate.index:
			lines.append(f"- {dept} : {rate[dept]}% ({int(left.get(dept, 0))} départs / {int(total[dept])} employés)")
		return "\n".join(lines)

	def headcount_by_role(self) -> str:
		if not self.is_ready:
			return "Dataset non chargé."
		df = self._df
		latest = df[df["SnapshotYear"] == df["SnapshotYear"].max()] if "SnapshotYear" in df.columns else df
		counts = latest.groupby("JobRole").size().sort_values(ascending=False)
		lines = [f"**Effectifs par rôle (année {latest['SnapshotYear'].max() if 'SnapshotYear' in latest.columns else 'N/A'}) :**"]
		for role, n in counts.items():
			lines.append(f"- {role} : {n}")
		lines.append(f"\n**Total : {counts.sum()} employés**")
		return "\n".join(lines)

	def absence_summary(self) -> str:
		if not self.is_ready:
			return "Dataset non chargé."
		df = self._df
		if "AbsenceDaysLast12Months" not in df.columns:
			return "Colonne AbsenceDaysLast12Months absente du dataset."
		avg = df.groupby("Department")["AbsenceDaysLast12Months"].mean().round(1)
		by_type = df.groupby("MainAbsenceType").size().sort_values(ascending=False)
		lines = ["**Résumé des absences :**", "", "*Moyenne de jours d'absence par département :*"]
		for dept, val in avg.items():
			lines.append(f"- {dept} : {val} jours/an")
		lines.append("\n*Répartition par type d'absence :*")
		for atype, n in by_type.items():
			lines.append(f"- {atype} : {n} employés")
		return "\n".join(lines)

	def salary_stats(self) -> str:
		if not self.is_ready:
			return "Dataset non chargé."
		df = self._df
		stats = df.groupby("Department")["MonthlyIncome"].agg(["mean", "min", "max"]).round(0)
		lines = ["**Statistiques salariales par département :**"]
		for dept in stats.index:
			lines.append(
				f"- {dept} : moy. {int(stats.loc[dept, 'mean'])}$ | "
				f"min {int(stats.loc[dept, 'min'])}$ | max {int(stats.loc[dept, 'max'])}$"
			)
		return "\n".join(lines)

	def contract_distribution(self) -> str:
		if not self.is_ready:
			return "Dataset non chargé."
		df = self._df
		if "ContractType" not in df.columns:
			return "Colonne ContractType absente."
		dist = df.groupby(["Department", "ContractType"]).size().unstack(fill_value=0)
		lines = ["**Répartition CDI/CDD par département :**"]
		for dept in dist.index:
			parts = [f"{ct}: {int(dist.loc[dept, ct])}" for ct in dist.columns]
			lines.append(f"- {dept} : {' | '.join(parts)}")
		return "\n".join(lines)

	def department_list(self) -> str:
		"""Liste des départements avec effectifs."""
		if not self.is_ready:
			return "Dataset non chargé."
		df = self._df
		latest = df[df["SnapshotYear"] == df["SnapshotYear"].max()] if "SnapshotYear" in df.columns else df
		counts = latest.groupby("Department").size().sort_values(ascending=False)
		lines = [f"**Il y a {len(counts)} départements :**"]
		for dept, n in counts.items():
			lines.append(f"- {dept} ({n} employés)")
		lines.append(f"\n**Total : {counts.sum()} employés**")
		return "\n".join(lines) 

	def seniority_distribution(self) -> str:
		"""Répartition des niveaux de séniorité par département."""
		if not self.is_ready:
			return "Dataset non chargé."
		df = self._df
		if "Seniority" not in df.columns:
			return "Colonne Seniority absente du dataset."
		latest = df[df["SnapshotYear"] == df["SnapshotYear"].max()] if "SnapshotYear" in df.columns else df
		lines = ["**Répartition de la séniorité par département :**"]
		for dept in sorted(latest["Department"].unique()):
			sub = latest[latest["Department"] == dept]
			dist = sub["Seniority"].value_counts()
			parts = [f"{s}: {int(dist.get(s, 0))}" for s in ["Junior", "Confirme", "Senior", "Expert"]]
			lines.append(f"- {dept} : {' | '.join(parts)}")
		return "\n".join(lines)

	def run_report(self, message: str) -> str:
		"""Choix automatique du rapport selon le message utilisateur."""
		text = message.lower()
		# Questions sur les départements (combien, liste, quels)
		if any(k in text for k in ["département", "departement", "départments", "departements"]):
			if any(k in text for k in ["attrition", "départ", "turnover"]):
				return self.attrition_by_department()
			if any(k in text for k in ["salaire", "rémunération", "paie"]):
				return self.salary_stats()
			if any(k in text for k in ["absence", "congé", "maladie"]):
				return self.absence_summary()
			if any(k in text for k in ["contrat", "cdi", "cdd"]):
				return self.contract_distribution()
			return self.department_list()
		if any(k in text for k in ["attrition", "départ", "turnover", "quitter"]):
			return self.attrition_by_department()
		if any(k in text for k in ["effectif", "headcount", "rôle", "role", "combien d'employé"]):
			return self.headcount_by_role()
		if any(k in text for k in ["absence", "congé", "maladie", "arrêt"]):
			return self.absence_summary()
		if any(k in text for k in ["salaire", "rémunération", "paie", "income", "revenu"]):
			return self.salary_stats()
		if any(k in text for k in ["contrat", "cdi", "cdd"]):
			return self.contract_distribution()
		if any(k in text for k in ["séniorité", "seniorite", "junior", "senior", "expert", "confirmé"]):
			return self.seniority_distribution()
		# Rapport par défaut : vue globale
		parts = [
			self.headcount_by_role(),
			self.attrition_by_department(),
			self.absence_summary(),
		]
		return "\n\n---\n\n".join(parts)

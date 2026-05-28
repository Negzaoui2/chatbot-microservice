from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.config.settings import get_settings


class PlanningService:
	"""Suggestions de planification / staffing basées sur le dataset HR."""

	def __init__(self) -> None:
		self._df: pd.DataFrame | None = None
		settings = get_settings()
		if settings.dataset_path and Path(settings.dataset_path).exists():
			self._df = pd.read_csv(settings.dataset_path)
			# Garder uniquement le snapshot le plus récent par employé
			if "SnapshotYear" in self._df.columns:
				idx = self._df.groupby("EmployeeNumber")["SnapshotYear"].idxmax()
				self._latest = self._df.loc[idx].copy()
			else:
				self._latest = self._df.copy()
			# Employés encore en poste
			self._active = self._latest[self._latest["Attrition"] == "No"].copy()
			print(f"[PLANNING] {len(self._active)} employés actifs chargés.")

	@property
	def is_ready(self) -> bool:
		return self._df is not None and not self._df.empty

	def find_available(self, message: str) -> str:
		"""Trouve des employés disponibles selon critères extraits du message."""
		if not self.is_ready:
			return "Dataset non chargé."
		df = self._active.copy()
		text = message.lower()

		# Filtrer par département si mentionné
		dept_map = {
			"dsi": "DSI",
			"web": "Developpement Web & Mobile",
			"mobile": "Developpement Web & Mobile",
			"frontend": "Developpement Web & Mobile",
			"backend": "Developpement Backend",
			"java": "Developpement Backend",
			"bi": "Business Intelligence & Data",
			"data": "Business Intelligence & Data",
			"ia": "Data Science & IA",
			"machine learning": "Data Science & IA",
			"data science": "Data Science & IA",
			"sap": "SAP & ERP",
			"erp": "SAP & ERP",
			"prodops": "ProdOps / Exploitation",
			"exploitation": "ProdOps / Exploitation",
			"infra": "ProdOps / Exploitation",
			"rh": "RH / Paie / SIRH",
			"sirh": "RH / Paie / SIRH",
			"paie": "RH / Paie / SIRH",
			"qa": "QA / Test & Recette",
			"test": "QA / Test & Recette",
			"recette": "QA / Test & Recette",
			"devops": "DevOps & Cloud",
			"cloud": "DevOps & Cloud",
			"pmo": "Gestion de Projet / PMO",
			"scrum": "Gestion de Projet / PMO",
			"projet": "Gestion de Projet / PMO",
		}
		for keyword, dept in dept_map.items():
			if keyword in text:
				df = df[df["Department"] == dept]
				break

		# Filtrer par rôle si mentionné
		role_keywords = {
			"développeur": "Developpeur",
			"developpeur": "Developpeur",
			"lead dev": "Lead Developpeur",
			"architecte": "Architecte",
			"consultant": "Consultant",
			"ingénieur": "Ingenieur",
			"ingenieur": "Ingenieur",
			"scrum master": "Scrum Master",
			"chef de projet": "Chef de Projet",
			"product owner": "Product Owner",
			"data scientist": "Data Scientist",
			"testeur": "Testeur",
			"devops": "DevOps",
			"sre": "SRE",
		}
		for keyword, role in role_keywords.items():
			if keyword in text:
				df = df[df["JobRole"].str.contains(role, case=False, na=False)]
				break

		# Prioriser ceux avec peu d'absences et bonne performance
		if "AbsenceDaysLast12Months" in df.columns:
			df = df.sort_values(["AbsenceDaysLast12Months", "PerformanceRating"], ascending=[True, False])

		top = df.head(10)
		if top.empty:
			return "Aucun employé disponible ne correspond aux critères."

		lines = [f"**{len(top)} employés disponibles trouvés :**", ""]
		for _, row in top.iterrows():
			absence = f", {int(row['AbsenceDaysLast12Months'])}j abs." if "AbsenceDaysLast12Months" in row.index else ""
			seniority = f" | {row['Seniority']}" if "Seniority" in row.index else ""
			lines.append(
				f"- **#{int(row['EmployeeNumber'])}** | {row['JobRole']} | "
				f"{row['Department']}{seniority} | Perf: {int(row['PerformanceRating'])}/4 | "
				f"{row['ContractType']}{absence}"
			)
		return "\n".join(lines)

	def optimize_workload(self, message: str) -> str:
		"""Suggère des optimisations de charge basées sur les données."""
		if not self.is_ready:
			return "Dataset non chargé."
		df = self._active.copy()
		text = message.lower()

		lines = ["**Suggestions d'optimisation de la charge :**", ""]

		# 1. Départements en sous-effectif (fort taux d'attrition)
		all_df = self._latest
		dept_attrition = all_df.groupby("Department")["Attrition"].apply(
			lambda x: (x == "Yes").sum() / len(x) * 100
		).round(1)
		risky = dept_attrition[dept_attrition > 15]
		if not risky.empty:
			lines.append("**Départements à risque (attrition > 15%) :**")
			for dept, rate in risky.items():
				active_count = len(df[df["Department"] == dept])
				lines.append(f"  - {dept} : {rate}% d'attrition, {active_count} actifs → **renforcer**")
			lines.append("")

		# 2. Employés surchargés (beaucoup d'heures sup)
		overtime = df[df["OverTime"] == "Yes"]
		if not overtime.empty:
			by_dept = overtime.groupby("Department").size()
			lines.append("**Employés en heures sup par département :**")
			for dept, n in by_dept.items():
				total = len(df[df["Department"] == dept])
				pct = round(n / total * 100, 1)
				lines.append(f"  - {dept} : {n}/{total} ({pct}%) en OverTime")
			lines.append("")

		# 3. Absences élevées
		if "AbsenceDaysLast12Months" in df.columns:
			high_abs = df[df["AbsenceDaysLast12Months"] > 20]
			if not high_abs.empty:
				lines.append(f"**{len(high_abs)} employés avec > 20 jours d'absence** (planifier des remplacements)")
				by_dept = high_abs.groupby("Department").size()
				for dept, n in by_dept.items():
					lines.append(f"  - {dept} : {n} employé(s)")
				lines.append("")

		# 4. Recommandations
		lines.append("**Recommandations :**")
		if not risky.empty:
			lines.append("- Lancer un plan de rétention ciblé pour les départements à risque")
		if not overtime.empty:
			lines.append("- Redistribuer la charge des employés en OverTime")
		if "AbsenceDaysLast12Months" in df.columns and len(df[df["AbsenceDaysLast12Months"] > 20]) > 0:
			lines.append("- Prévoir des remplacements temporaires pour les absences longues")
		lines.append("- Considérer des recrutements CDD pour les pics de charge")

		return "\n".join(lines)

	def suggest(self, message: str) -> str:
		"""Point d'entrée : choisit entre disponibilité et optimisation."""
		text = message.lower()
		if any(k in text for k in ["disponible", "dispo", "libre", "affecter", "projet", "qui peut"]):
			return self.find_available(message)
		if any(k in text for k in ["optimise", "charge", "redistribu", "surcharge", "répartir"]):
			return self.optimize_workload(message)
		# Par défaut : les deux
		return self.find_available(message) + "\n\n---\n\n" + self.optimize_workload(message)

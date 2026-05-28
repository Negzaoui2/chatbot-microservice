from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from app.config.settings import get_settings


class ExportService:
	"""Génère des rapports PDF et Excel à partir du dataset HR."""

	def __init__(self) -> None:
		self._df: pd.DataFrame | None = None
		settings = get_settings()
		if settings.dataset_path and Path(settings.dataset_path).exists():
			self._df = pd.read_csv(settings.dataset_path)

	@property
	def is_ready(self) -> bool:
		return self._df is not None and not self._df.empty

	def _get_report_data(self, report_type: str) -> dict[str, pd.DataFrame]:
		"""Retourne les DataFrames pour le rapport demandé."""
		df = self._df
		if report_type == "attrition":
			total = df.groupby("Department")["Attrition"].count().rename("Total")
			left = df[df["Attrition"] == "Yes"].groupby("Department")["Attrition"].count().rename("Départs")
			rate = ((left / total) * 100).fillna(0).round(1).rename("Taux (%)")
			result = pd.concat([total, left.fillna(0).astype(int), rate], axis=1).reset_index()
			return {"Attrition par département": result}

		elif report_type == "headcount":
			latest = df[df["SnapshotYear"] == df["SnapshotYear"].max()] if "SnapshotYear" in df.columns else df
			counts = latest.groupby("JobRole").size().reset_index(name="Effectif").sort_values("Effectif", ascending=False)
			return {"Effectifs par rôle": counts}

		elif report_type == "absence":
			avg = df.groupby("Department")["AbsenceDaysLast12Months"].mean().round(1).reset_index()
			avg.columns = ["Département", "Moyenne jours absence"]
			by_type = df.groupby("MainAbsenceType").size().reset_index(name="Nombre employés")
			by_type.columns = ["Type absence", "Nombre employés"]
			return {"Absences par département": avg, "Absences par type": by_type}

		elif report_type == "salary":
			stats = df.groupby("Department")["MonthlyIncome"].agg(["mean", "min", "max"]).round(0).astype(int).reset_index()
			stats.columns = ["Département", "Salaire moyen ($)", "Min ($)", "Max ($)"]
			return {"Statistiques salariales": stats}

		elif report_type == "contract":
			dist = df.groupby(["Department", "ContractType"]).size().unstack(fill_value=0).reset_index()
			return {"Répartition contrats": dist}

		else:
			# Rapport global
			total = df.groupby("Department")["Attrition"].count().rename("Total")
			left = df[df["Attrition"] == "Yes"].groupby("Department")["Attrition"].count().rename("Départs")
			rate = ((left / total) * 100).fillna(0).round(1).rename("Taux (%)")
			attrition = pd.concat([total, left.fillna(0).astype(int), rate], axis=1).reset_index()

			latest = df[df["SnapshotYear"] == df["SnapshotYear"].max()] if "SnapshotYear" in df.columns else df
			headcount = latest.groupby("JobRole").size().reset_index(name="Effectif").sort_values("Effectif", ascending=False)

			stats = df.groupby("Department")["MonthlyIncome"].agg(["mean", "min", "max"]).round(0).astype(int).reset_index()
			stats.columns = ["Département", "Salaire moyen ($)", "Min ($)", "Max ($)"]

			return {
				"Attrition": attrition,
				"Effectifs": headcount,
				"Salaires": stats,
			}

	def export_excel(self, report_type: str = "global") -> io.BytesIO:
		"""Génère un fichier Excel en mémoire avec les données du rapport."""
		sheets = self._get_report_data(report_type)
		buffer = io.BytesIO()
		with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
			for sheet_name, dataframe in sheets.items():
				dataframe.to_excel(writer, sheet_name=sheet_name[:31], index=False)
		buffer.seek(0)
		return buffer

	def export_pdf(self, report_type: str = "global") -> io.BytesIO:
		"""Génère un fichier PDF en mémoire avec les données du rapport."""
		from reportlab.lib import colors
		from reportlab.lib.pagesizes import A4, landscape
		from reportlab.lib.styles import getSampleStyleSheet
		from reportlab.lib.units import cm
		from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

		sheets = self._get_report_data(report_type)
		buffer = io.BytesIO()
		doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), leftMargin=1 * cm, rightMargin=1 * cm)
		styles = getSampleStyleSheet()
		elements = []

		title = f"Rapport HR - {report_type.capitalize()}"
		elements.append(Paragraph(title, styles["Title"]))
		elements.append(Spacer(1, 0.5 * cm))

		for sheet_name, dataframe in sheets.items():
			elements.append(Paragraph(sheet_name, styles["Heading2"]))
			elements.append(Spacer(1, 0.3 * cm))

			# Construire le tableau
			data = [list(dataframe.columns)] + dataframe.values.tolist()
			table = Table(data, repeatRows=1)
			table.setStyle(TableStyle([
				("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
				("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
				("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
				("FONTSIZE", (0, 0), (-1, 0), 9),
				("FONTSIZE", (0, 1), (-1, -1), 8),
				("ALIGN", (0, 0), (-1, -1), "CENTER"),
				("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
				("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#D9E2F3")]),
			]))
			elements.append(table)
			elements.append(Spacer(1, 0.8 * cm))

		doc.build(elements)
		buffer.seek(0)
		return buffer

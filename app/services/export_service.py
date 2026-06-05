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
		"""Génère un fichier PDF professionnel en mémoire."""
		from datetime import datetime

		from reportlab.lib import colors
		from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
		from reportlab.lib.pagesizes import A4, landscape
		from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
		from reportlab.lib.units import cm, mm
		from reportlab.platypus import (
			SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
			HRFlowable, Frame, PageTemplate, BaseDocTemplate,
		)

		# Couleurs de la charte
		PRIMARY = colors.HexColor("#1B2A4A")      # Bleu foncé
		ACCENT = colors.HexColor("#2E86AB")       # Bleu vif
		LIGHT_BG = colors.HexColor("#F4F7FA")     # Gris très clair
		ROW_ALT = colors.HexColor("#EAF2F8")      # Bleu pâle alterné
		HEADER_BG = colors.HexColor("#1B2A4A")    # Header tableau
		TEXT_DARK = colors.HexColor("#2C3E50")     # Texte principal

		sheets = self._get_report_data(report_type)
		buffer = io.BytesIO()

		# Marges et document
		doc = SimpleDocTemplate(
			buffer,
			pagesize=landscape(A4),
			leftMargin=2 * cm,
			rightMargin=2 * cm,
			topMargin=2.5 * cm,
			bottomMargin=2 * cm,
		)

		# Styles personnalisés
		styles = getSampleStyleSheet()
		style_title = ParagraphStyle(
			"CustomTitle",
			parent=styles["Title"],
			fontSize=22,
			textColor=PRIMARY,
			spaceAfter=6,
			fontName="Helvetica-Bold",
		)
		style_subtitle = ParagraphStyle(
			"CustomSubtitle",
			parent=styles["Normal"],
			fontSize=11,
			textColor=colors.HexColor("#7F8C8D"),
			spaceAfter=20,
			fontName="Helvetica",
		)
		style_section = ParagraphStyle(
			"SectionTitle",
			parent=styles["Heading2"],
			fontSize=14,
			textColor=ACCENT,
			spaceBefore=16,
			spaceAfter=8,
			fontName="Helvetica-Bold",
			borderPadding=(0, 0, 4, 0),
		)
		style_footer = ParagraphStyle(
			"Footer",
			parent=styles["Normal"],
			fontSize=8,
			textColor=colors.HexColor("#95A5A6"),
			alignment=TA_CENTER,
		)

		elements = []

		# ── Header / Titre ─────────────────────────────────────────────────
		now = datetime.now().strftime("%d/%m/%Y à %H:%M")
		report_titles = {
			"global": "Rapport Global RH",
			"attrition": "Rapport d'Attrition",
			"headcount": "Rapport des Effectifs",
			"absence": "Rapport des Absences",
			"salary": "Rapport Salarial",
			"contract": "Rapport des Contrats",
		}
		title_text = report_titles.get(report_type, f"Rapport {report_type.capitalize()}")

		elements.append(Paragraph("📊 Plateforme de Staffing — Rapport HR", style_title))
		elements.append(Paragraph(f"{title_text} • Généré le {now}", style_subtitle))

		# Ligne de séparation
		elements.append(HRFlowable(
			width="100%", thickness=2, color=ACCENT,
			spaceBefore=2, spaceAfter=16,
		))

		# ── Contenu : tableaux ─────────────────────────────────────────────
		for sheet_name, dataframe in sheets.items():
			elements.append(Paragraph(f"■ {sheet_name}", style_section))

			# Préparer les données du tableau
			col_headers = [str(c) for c in dataframe.columns]
			data = [col_headers]
			for _, row in dataframe.iterrows():
				data.append([str(v) for v in row.values])

			# Calculer les largeurs de colonnes proportionnelles
			available_width = landscape(A4)[0] - 4 * cm
			n_cols = len(col_headers)
			col_widths = [available_width / n_cols] * n_cols

			table = Table(data, colWidths=col_widths, repeatRows=1)
			table.setStyle(TableStyle([
				# Header
				("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
				("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
				("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
				("FONTSIZE", (0, 0), (-1, 0), 9),
				("BOTTOMPADDING", (0, 0), (-1, 0), 10),
				("TOPPADDING", (0, 0), (-1, 0), 10),
				# Corps
				("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
				("FONTSIZE", (0, 1), (-1, -1), 8.5),
				("TOPPADDING", (0, 1), (-1, -1), 7),
				("BOTTOMPADDING", (0, 1), (-1, -1), 7),
				# Alignement
				("ALIGN", (0, 0), (-1, -1), "CENTER"),
				("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
				# Bordures légères
				("LINEBELOW", (0, 0), (-1, 0), 1.5, ACCENT),
				("LINEBELOW", (0, 1), (-1, -2), 0.5, colors.HexColor("#DCE4EC")),
				("LINEBELOW", (0, -1), (-1, -1), 1, PRIMARY),
				# Alternance de couleurs
				("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW_ALT]),
			]))
			elements.append(table)
			elements.append(Spacer(1, 1 * cm))

		# ── Footer ─────────────────────────────────────────────────────────
		elements.append(Spacer(1, 1 * cm))
		elements.append(HRFlowable(
			width="100%", thickness=0.5, color=colors.HexColor("#BDC3C7"),
			spaceBefore=8, spaceAfter=8,
		))
		elements.append(Paragraph(
			f"Document généré automatiquement par le Chatbot HR — {now} • Confidentiel",
			style_footer,
		))

		doc.build(elements)
		buffer.seek(0)
		return buffer

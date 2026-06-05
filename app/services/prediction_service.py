from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from app.config.settings import get_settings


class PredictionService:
	"""Service ML : prédit le risque d'attrition des employés (Random Forest)."""

	# Features utilisées pour l'entraînement
	NUMERIC_FEATURES = [
		"Age", "MonthlyIncome", "DistanceFromHome",
		"YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
		"NumCompaniesWorked", "TotalWorkingYears",
		"JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance",
		"JobInvolvement", "PerformanceRating",
		"AbsenceDaysLast12Months", "TrainingTimesLastYear",
	]

	CATEGORICAL_FEATURES = [
		"OverTime", "Department", "JobRole", "Seniority", "ContractType",
	]

	def __init__(self) -> None:
		self._model: RandomForestClassifier | None = None
		self._encoders: dict[str, LabelEncoder] = {}
		self._df: pd.DataFrame | None = None
		self._accuracy: float = 0.0
		self._feature_importances: dict[str, float] = {}

		settings = get_settings()
		if settings.dataset_path and Path(settings.dataset_path).exists():
			self._df = pd.read_csv(settings.dataset_path)
			self._train()

	@property
	def is_ready(self) -> bool:
		return self._model is not None

	def _train(self) -> None:
		"""Entraîne le modèle Random Forest sur le dataset."""
		df = self._df.copy()

		# Encoder la target
		df["Attrition_encoded"] = (df["Attrition"] == "Yes").astype(int)

		# Préparer les features
		features_df = self._prepare_features(df)
		if features_df is None:
			return

		X = features_df
		y = df["Attrition_encoded"]

		# Split train/test
		X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size=0.2, random_state=42, stratify=y,
		)

		# Entraîner le modèle
		self._model = RandomForestClassifier(
			n_estimators=100,
			max_depth=12,
			min_samples_split=5,
			random_state=42,
			class_weight="balanced",  # Gère le déséquilibre Yes/No
			n_jobs=-1,
		)
		self._model.fit(X_train, y_train)

		# Évaluation
		y_pred = self._model.predict(X_test)
		self._accuracy = accuracy_score(y_test, y_pred)

		# Importances des features
		importances = self._model.feature_importances_
		feature_names = list(X.columns)
		self._feature_importances = dict(
			sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
		)

		print(f"[PREDICTION] Modèle entraîné — Accuracy: {self._accuracy:.2%}")
		print(f"[PREDICTION] Top 5 features: {list(self._feature_importances.keys())[:5]}")

	def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame | None:
		"""Prépare le DataFrame de features (encodage + sélection)."""
		features = pd.DataFrame(index=df.index)

		# Features numériques
		for col in self.NUMERIC_FEATURES:
			if col in df.columns:
				features[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

		# Features catégorielles (Label Encoding)
		for col in self.CATEGORICAL_FEATURES:
			if col in df.columns:
				if col not in self._encoders:
					self._encoders[col] = LabelEncoder()
					features[col] = self._encoders[col].fit_transform(df[col].astype(str))
				else:
					# Pour la prédiction, gérer les valeurs inconnues
					known = set(self._encoders[col].classes_)
					features[col] = df[col].astype(str).apply(
						lambda x: self._encoders[col].transform([x])[0] if x in known else -1
					)

		if features.empty:
			print("[PREDICTION] Aucune feature valide trouvée.")
			return None

		return features

	def predict_at_risk(self, top_n: int = 10, department: str | None = None) -> str:
		"""Prédit les employés à risque d'attrition."""
		if not self.is_ready:
			return "Modèle de prédiction non disponible."

		df = self._df.copy()

		# Filtrer par département si demandé
		if department:
			df_filtered = df[df["Department"].str.lower().str.contains(department.lower())]
			if df_filtered.empty:
				return f"Aucun employé trouvé dans le département '{department}'."
			df = df_filtered

		# Ne garder que les employés encore présents (Attrition = No)
		df_active = df[df["Attrition"] == "No"].copy()
		if df_active.empty:
			return "Aucun employé actif trouvé."

		# Préparer les features pour la prédiction
		features = self._prepare_features(df_active)
		if features is None:
			return "Erreur lors de la préparation des features."

		# Prédire les probabilités
		probas = self._model.predict_proba(features)[:, 1]  # Proba de la classe 1 (Yes)
		df_active = df_active.copy()
		df_active["risk_score"] = (probas * 100).round(1)

		# Trier par score de risque décroissant
		top_risk = df_active.nlargest(top_n, "risk_score")

		# Formater la réponse
		lines = [
			f"**🚨 Top {len(top_risk)} employés à risque d'attrition :**",
			f"_(Modèle Random Forest — Précision : {self._accuracy:.1%})_\n",
		]

		for i, (_, emp) in enumerate(top_risk.iterrows(), 1):
			risk = emp["risk_score"]
			risk_icon = "🔴" if risk >= 70 else "🟠" if risk >= 50 else "🟡"
			lines.append(
				f"{i}. {risk_icon} **Employé #{int(emp['EmployeeNumber'])}** — "
				f"Risque : {risk}%\n"
				f"   - Poste : {emp.get('JobRole', 'N/A')} | "
				f"Département : {emp.get('Department', 'N/A')}\n"
				f"   - Ancienneté : {int(emp.get('YearsAtCompany', 0))} ans | "
				f"Satisfaction : {int(emp.get('JobSatisfaction', 0))}/4 | "
				f"Absences : {int(emp.get('AbsenceDaysLast12Months', 0))} jours"
			)

		# Ajouter les facteurs clés
		lines.append("\n**📊 Facteurs de risque principaux (importance du modèle) :**")
		for feat, imp in list(self._feature_importances.items())[:5]:
			lines.append(f"- {feat} : {imp:.1%}")

		return "\n".join(lines)

	def get_model_info(self) -> str:
		"""Retourne les infos sur le modèle entraîné."""
		if not self.is_ready:
			return "Modèle non entraîné."

		lines = [
			"**📈 Informations sur le modèle de prédiction d'attrition :**\n",
			f"- Algorithme : Random Forest (100 arbres)",
			f"- Précision (accuracy) : {self._accuracy:.1%}",
			f"- Nombre de features : {len(self.NUMERIC_FEATURES) + len(self.CATEGORICAL_FEATURES)}",
			f"- Dataset : {len(self._df)} enregistrements",
			f"\n**Top 10 features les plus importantes :**",
		]
		for feat, imp in list(self._feature_importances.items())[:10]:
			bar = "█" * int(imp * 50)
			lines.append(f"- {feat} : {imp:.1%} {bar}")

		return "\n".join(lines)

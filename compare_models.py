"""
Comparaison de modèles ML pour la prédiction d'attrition.
- Random Forest (Bagging)
- Gradient Boosting (Boosting)

Exécuter : python compare_models.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, precision_score

from app.config.settings import get_settings


def load_and_prepare():
	"""Charge le dataset et prépare les features."""
	settings = get_settings()
	df = pd.read_csv(settings.dataset_path)

	# Target
	df["Attrition_encoded"] = (df["Attrition"] == "Yes").astype(int)

	# Features numériques
	numeric_features = [
		"Age", "MonthlyIncome", "DistanceFromHome",
		"YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
		"NumCompaniesWorked", "TotalWorkingYears",
		"JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance",
		"JobInvolvement", "PerformanceRating",
		"AbsenceDaysLast12Months", "TrainingTimesLastYear",
	]

	# Features catégorielles
	categorical_features = [
		"OverTime", "Department", "JobRole", "Seniority", "ContractType",
	]

	features = pd.DataFrame(index=df.index)

	for col in numeric_features:
		if col in df.columns:
			features[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

	encoders = {}
	for col in categorical_features:
		if col in df.columns:
			encoders[col] = LabelEncoder()
			features[col] = encoders[col].fit_transform(df[col].astype(str))

	X = features
	y = df["Attrition_encoded"]

	return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def compare():
	"""Compare Random Forest vs Gradient Boosting."""
	X_train, X_test, y_train, y_test = load_and_prepare()

	print("=" * 70)
	print("   COMPARAISON DES MODÈLES — Prédiction d'Attrition")
	print("=" * 70)
	print(f"\n   Dataset : {len(X_train) + len(X_test)} employés")
	print(f"   Train : {len(X_train)} | Test : {len(X_test)}")
	print(f"   Features : {X_train.shape[1]}")
	print(f"   Distribution target : {dict(y_test.value_counts())}")

	# ── Modèle 1 : Random Forest ──────────────────────────────────────
	print("\n" + "─" * 70)
	print("  🌲 MODÈLE 1 : Random Forest (Bagging)")
	print("─" * 70)

	rf = RandomForestClassifier(
		n_estimators=100,
		max_depth=12,
		min_samples_split=5,
		random_state=42,
		class_weight="balanced",
		n_jobs=-1,
	)
	rf.fit(X_train, y_train)
	y_pred_rf = rf.predict(X_test)

	acc_rf = accuracy_score(y_test, y_pred_rf)
	f1_rf = f1_score(y_test, y_pred_rf)
	recall_rf = recall_score(y_test, y_pred_rf)
	precision_rf = precision_score(y_test, y_pred_rf)

	print(f"\n  Accuracy  : {acc_rf:.4f} ({acc_rf:.2%})")
	print(f"  F1-Score  : {f1_rf:.4f}")
	print(f"  Recall    : {recall_rf:.4f}")
	print(f"  Precision : {precision_rf:.4f}")
	print(f"\n{classification_report(y_test, y_pred_rf, target_names=['Reste (No)', 'Part (Yes)'])}")

	# ── Modèle 2 : Gradient Boosting ──────────────────────────────────
	print("─" * 70)
	print("  🚀 MODÈLE 2 : Gradient Boosting (Boosting)")
	print("─" * 70)

	gb = GradientBoostingClassifier(
		n_estimators=150,
		max_depth=5,
		learning_rate=0.1,
		min_samples_split=5,
		random_state=42,
	)
	gb.fit(X_train, y_train)
	y_pred_gb = gb.predict(X_test)

	acc_gb = accuracy_score(y_test, y_pred_gb)
	f1_gb = f1_score(y_test, y_pred_gb)
	recall_gb = recall_score(y_test, y_pred_gb)
	precision_gb = precision_score(y_test, y_pred_gb)

	print(f"\n  Accuracy  : {acc_gb:.4f} ({acc_gb:.2%})")
	print(f"  F1-Score  : {f1_gb:.4f}")
	print(f"  Recall    : {recall_gb:.4f}")
	print(f"  Precision : {precision_gb:.4f}")
	print(f"\n{classification_report(y_test, y_pred_gb, target_names=['Reste (No)', 'Part (Yes)'])}")

	# ── Tableau comparatif ─────────────────────────────────────────────
	print("=" * 70)
	print("   📊 TABLEAU COMPARATIF")
	print("=" * 70)
	print(f"\n  {'Métrique':<15} {'Random Forest':<18} {'Gradient Boosting':<18} {'Meilleur'}")
	print(f"  {'─'*15} {'─'*18} {'─'*18} {'─'*15}")

	metrics = [
		("Accuracy", acc_rf, acc_gb),
		("F1-Score", f1_rf, f1_gb),
		("Recall", recall_rf, recall_gb),
		("Precision", precision_rf, precision_gb),
	]

	for name, rf_val, gb_val in metrics:
		winner = "🌲 Random Forest" if rf_val >= gb_val else "🚀 Gradient Boost"
		print(f"  {name:<15} {rf_val:<18.4f} {gb_val:<18.4f} {winner}")

	# ── Conclusion ─────────────────────────────────────────────────────
	print(f"\n{'=' * 70}")
	rf_score = sum([acc_rf, f1_rf, recall_rf, precision_rf])
	gb_score = sum([acc_gb, f1_gb, recall_gb, precision_gb])

	if rf_score >= gb_score:
		print("  ✅ CONCLUSION : Random Forest est le meilleur modèle pour ce dataset.")
	else:
		print("  ✅ CONCLUSION : Gradient Boosting est le meilleur modèle pour ce dataset.")
	print("=" * 70)


if __name__ == "__main__":
	compare()

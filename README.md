# Chatbot Microservice (Python)

Microservice FastAPI pour un assistant RH intelligent (PFE – Plateforme de planification de staffing) avec **RAG** basé sur le dataset **HR-Employee-Attrition** (Kaggle, format `.xlsb`).

## Dataset

- **Nom** : HR-Employee-Attrition.xlsb
- **Colonnes** : Age, Attrition, BusinessTravel, DailyRate, Departement, DistanceFromHome, Education, EducationField, EmployeeCount, EmployeeNumber, EnvironnmentStatisfaction, Gender, HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked, Over18, OverTime, PercentSalaryHike, PerformanceRating, RelationShipSatisfaction, StandardHours, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager

## Démarrage

1) Installer les dépendances

```bash
pip install -r requirements.txt
```

2) Configurer `.env` (déjà créé, à adapter)

```env
DATASET_PATH=C:\chemin\vers\HR-Employee-Attrition.xlsb
DATASET_TEXT_COLUMNS=Attrition,BusinessTravel,Departement,EducationField,Gender,JobRole,MaritalStatus,OverTime
```

3) Lancer le serveur

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /health`
- `POST /chat`

Payload exemple:

```json
{ "message": "Quels sont les types de congés disponibles ?" }
```

# prepare_dataset.py
# Transforme le dataset Kaggle HR-Employee-Attrition en un dataset
# realiste pour une SSII / ESN (type SopraHR).
import pandas as pd
import numpy as np
from pathlib import Path
import random

random.seed(42)
np.random.seed(42)

DATA_DIR = Path(__file__).parent / "data"
INPUT_FILE = DATA_DIR / "HR-Employee-Attrition.csv"
OUTPUT_FILE = DATA_DIR / "HR-Enriched-History-With-IT.csv"

df = pd.read_csv(INPUT_FILE)
print("Dataset original charge :", df.shape)

# == DEPARTEMENTS IT (11 departements SopraHR, SANS doublons) ==
IT_DEPARTMENTS = [
    "DSI",
    "Developpement Web & Mobile",
    "Developpement Backend",
    "Business Intelligence & Data",
    "Data Science & IA",
    "SAP & ERP",
    "ProdOps / Exploitation",
    "RH / Paie / SIRH",
    "QA / Test & Recette",
    "DevOps & Cloud",
    "Gestion de Projet / PMO",
]
DEPT_WEIGHTS = [0.08, 0.12, 0.16, 0.10, 0.09, 0.12, 0.08, 0.05, 0.08, 0.07, 0.05]

df["Department"] = np.random.choice(IT_DEPARTMENTS, size=len(df), p=DEPT_WEIGHTS)

# == JOB ROLES IT ==
ROLES_BY_DEPT = {
    "DSI": ["Architecte SI", "Responsable Securite SI", "Urbaniste SI", "Directeur Technique"],
    "Developpement Web & Mobile": ["Developpeur Frontend", "Developpeur Fullstack", "Developpeur Mobile", "Lead Developpeur Web"],
    "Developpement Backend": ["Developpeur Java", "Developpeur Python", "Developpeur .NET", "Lead Developpeur Backend"],
    "Business Intelligence & Data": ["Consultant BI", "Data Analyst", "Developpeur ETL", "Ingenieur Data"],
    "Data Science & IA": ["Data Scientist", "Ingenieur Machine Learning", "Data Engineer", "Ingenieur NLP"],
    "SAP & ERP": ["Consultant SAP", "Developpeur ABAP", "Consultant ERP", "Architecte SAP"],
    "ProdOps / Exploitation": ["Ingenieur de Production", "Administrateur Systemes", "Ingenieur Reseau", "Technicien Support N3"],
    "RH / Paie / SIRH": ["Consultant SIRH", "Developpeur SIRH", "Analyste Paie", "Chef de Projet SIRH"],
    "QA / Test & Recette": ["Ingenieur QA", "Testeur Automatisation", "Responsable Recette", "Ingenieur Test Performance"],
    "DevOps & Cloud": ["Ingenieur DevOps", "Ingenieur Cloud", "SRE", "Architecte Cloud"],
    "Gestion de Projet / PMO": ["Scrum Master", "Chef de Projet", "Product Owner", "Directeur de Programme"],
}

df["JobRole"] = df["Department"].apply(lambda dept: random.choice(ROLES_BY_DEPT.get(dept, ["Consultant"])))

# == MAIN DOMAIN (coherent avec departement) ==
DOMAIN_BY_DEPT = {
    "DSI": "Architecture & Securite SI",
    "Developpement Web & Mobile": "Developpement Fullstack JS / Web",
    "Developpement Backend": "Developpement Java / Backend",
    "Business Intelligence & Data": "Business Intelligence / Data Viz",
    "Data Science & IA": "Data Science / IA / Machine Learning",
    "SAP & ERP": "SAP / ERP Integration",
    "ProdOps / Exploitation": "Infrastructure & Production",
    "RH / Paie / SIRH": "SIRH / Paie",
    "QA / Test & Recette": "Test / QA / Automatisation",
    "DevOps & Cloud": "DevOps / Cloud / Infrastructure",
    "Gestion de Projet / PMO": "Gestion de Projet / Scrum",
}
df["MainDomain"] = df["Department"].map(DOMAIN_BY_DEPT)

# == SKILLS ==
SKILLS_BY_DEPT = {
    "DSI": ["TOGAF", "ArchiMate", "Cybersecurite", "ISO 27001", "RGPD", "IAM", "PKI", "Firewall"],
    "Developpement Web & Mobile": ["Angular", "React", "TypeScript", "Node.js", "HTML/CSS", "Flutter", "Swift", "REST API"],
    "Developpement Backend": ["Java", "Spring Boot", "Python", "Django", "C#", ".NET", "Hibernate", "REST API", "SQL", "Maven", "Git"],
    "Business Intelligence & Data": ["Power BI", "Tableau", "SQL", "ETL", "SSIS", "DAX", "Talend", "Data Modeling"],
    "Data Science & IA": ["Python", "Pandas", "Scikit-learn", "TensorFlow", "PyTorch", "NLP", "Machine Learning", "Deep Learning", "Spark"],
    "SAP & ERP": ["SAP ABAP", "SAP Fiori", "SAP HANA", "S/4HANA", "SAP PI/PO", "OData", "SAP BW"],
    "ProdOps / Exploitation": ["Linux", "Windows Server", "Nagios", "Shell Script", "ITIL", "VMware", "Cron", "Monitoring"],
    "RH / Paie / SIRH": ["SAP HCM", "Workday", "ADP", "Talentsoft", "SQL", "Paie", "Droit Social"],
    "QA / Test & Recette": ["Selenium", "Cypress", "Postman", "JUnit", "Robot Framework", "JMeter", "TestNG", "Cucumber"],
    "DevOps & Cloud": ["Docker", "Kubernetes", "AWS", "Azure", "GCP", "Terraform", "Ansible", "CI/CD", "Jenkins", "GitLab CI"],
    "Gestion de Projet / PMO": ["Scrum", "SAFe", "PMP", "Jira", "Confluence", "MS Project", "Agile", "Risk Management"],
}
CROSS_SKILLS = ["Git", "SQL", "Agile", "Jira", "Python", "REST API", "Docker", "Linux"]

def generate_skills(dept):
    base = SKILLS_BY_DEPT.get(dept, ["SQL", "Git"])
    extras = random.sample(CROSS_SKILLS, k=random.randint(1, 3))
    combined = list(set(base + extras))
    n = random.randint(3, 7)
    return ", ".join(random.sample(combined, k=min(n, len(combined))))

df["Skills"] = df["Department"].apply(generate_skills)

# == CONTRAT CDI/CDD (ratio realiste ~80% CDI) ==
cdd_proba = np.where(df["JobLevel"] <= 1, 0.40,
            np.where(df["JobLevel"] == 2, 0.20,
            np.where(df["JobLevel"] == 3, 0.10, 0.05)))
df["ContractType"] = np.where(np.random.random(len(df)) < cdd_proba, "CDD", "CDI")

# == EDUCATION FIELD -> filieres IT ==
IT_EDUCATION_FIELDS = [
    "Informatique", "Genie Logiciel", "Mathematiques & Statistiques",
    "Reseaux & Telecoms", "Systemes d Information", "Commerce / Management", "Autre",
]
EDU_WEIGHTS = [0.30, 0.20, 0.12, 0.10, 0.13, 0.10, 0.05]
df["EducationField"] = np.random.choice(IT_EDUCATION_FIELDS, size=len(df), p=EDU_WEIGHTS)

# == SENIORITY ==
def compute_seniority(total_years):
    if total_years <= 2:
        return "Junior"
    elif total_years <= 6:
        return "Confirme"
    elif total_years <= 12:
        return "Senior"
    else:
        return "Expert"

df["Seniority"] = df["TotalWorkingYears"].apply(compute_seniority)

# == CERTIFICATIONS ==
CERTS_BY_DEPT = {
    "DSI": ["TOGAF Certified", "CISSP", "ISO 27001 Lead Auditor", "CISA"],
    "Developpement Web & Mobile": ["Meta Front-End Developer", "Google Mobile Web Specialist", "AWS Developer Associate"],
    "Developpement Backend": ["Oracle Certified Java SE", "Spring Professional", "AWS Developer Associate", "Microsoft Azure Developer"],
    "Business Intelligence & Data": ["Microsoft PL-300 (Power BI)", "Tableau Desktop Specialist", "Google Data Analytics"],
    "Data Science & IA": ["TensorFlow Developer", "AWS ML Specialty", "Google Professional ML Engineer", "IBM Data Science"],
    "SAP & ERP": ["SAP Certified Application Associate", "SAP HANA 2.0", "SAP Fiori Certified"],
    "ProdOps / Exploitation": ["ITIL Foundation", "ITIL Intermediate", "VMware VCP", "Red Hat RHCSA"],
    "RH / Paie / SIRH": ["SAP HCM Certified", "Workday Pro", "ADP Certified"],
    "QA / Test & Recette": ["ISTQB Foundation", "ISTQB Advanced", "Selenium Certified", "CSTE"],
    "DevOps & Cloud": ["AWS Solutions Architect", "CKA (Kubernetes)", "Azure DevOps Engineer Expert", "Terraform Associate", "Docker Certified"],
    "Gestion de Projet / PMO": ["PMP", "PSM I (Scrum Master)", "PSM II", "SAFe Agilist", "PRINCE2"],
}

def generate_certifications(row):
    dept = row["Department"]
    seniority = row["Seniority"]
    pool = CERTS_BY_DEPT.get(dept, ["ITIL Foundation"])
    if seniority == "Junior":
        n = random.choice([0, 0, 1])
    elif seniority == "Confirme":
        n = random.choice([0, 1, 1, 2])
    else:
        n = random.choice([1, 1, 2, 2, 3])
    if n == 0:
        return "Aucune"
    return ", ".join(random.sample(pool, k=min(n, len(pool))))

df["Certifications"] = df.apply(generate_certifications, axis=1)

# == ABSENCES ==
df["AbsenceDaysLast12Months"] = np.random.randint(0, 45, len(df))
df["MainAbsenceType"] = np.select(
    [df["AbsenceDaysLast12Months"] > 20, df["AbsenceDaysLast12Months"].between(8, 20)],
    ["Maladie longue / Arret", "Conges / RTT / courts arrets"],
    default="Peu ou pas d absence"
)

# == ATTRITION REASON ==
def get_attrition_reason(row):
    if row["Attrition"] == "No":
        return "Toujours en poste"
    reasons = []
    if row["MonthlyIncome"] < 3500:
        reasons.append("salaire insuffisant")
    if row["PercentSalaryHike"] < 13:
        reasons.append("peu d augmentation")
    if row["YearsSinceLastPromotion"] > 4:
        reasons.append("pas de promotion recente")
    if row["WorkLifeBalance"] <= 2:
        reasons.append("mauvais equilibre vie pro/perso")
    if row["OverTime"] == "Yes" and row["JobSatisfaction"] <= 2:
        reasons.append("heures supplementaires excessives")
    if row["DistanceFromHome"] > 20:
        reasons.append("distance domicile-travail trop importante")
    return ", ".join(reasons[:3]) or "Autres raisons non identifiees"

df["AttritionReason"] = df.apply(get_attrition_reason, axis=1)

# == PROJECT HISTORY (coherent avec le domaine !) ==
PROJECT_TYPES = [
    "Migration", "Nouvelle implementation", "Optimisation",
    "Evolution", "Projet client", "Maintenance", "Refonte",
    "Audit", "POC", "Integration",
]
PROJECT_ROLES = {
    "Junior": ["Developpeur", "Testeur", "Analyste Junior"],
    "Confirme": ["Developpeur", "Analyste", "Consultant", "Ingenieur"],
    "Senior": ["Lead Dev", "Tech Lead", "Consultant Senior", "Architecte"],
    "Expert": ["Tech Lead", "Architecte", "Directeur Technique", "Scrum Master"],
}

def generate_project_history(row):
    domain = row["MainDomain"]
    seniority = row["Seniority"]
    n = random.randint(2, 6)
    years = sorted(random.sample(range(2019, 2026), min(n, 7)), reverse=True)
    roles_pool = PROJECT_ROLES.get(seniority, ["Developpeur"])
    projects = []
    for y in years:
        ptype = random.choice(PROJECT_TYPES)
        role = random.choice(roles_pool)
        projects.append(f"{domain} - {ptype} {y} ({role}, {y})")
    return "; ".join(projects)

df["ProjectHistory"] = df.apply(generate_project_history, axis=1)

# == NETTOYAGE ==
if "IT_Department" in df.columns:
    df.drop(columns=["IT_Department"], inplace=True)
DROP_COLS = ["EmployeeCount", "Over18", "StandardHours"]
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

# == HISTORIQUE ANNUEL (snapshots) ==
def add_history(row):
    n = random.randint(2, 5)
    rows_list = []
    for i in range(n):
        r = row.copy()
        r["SnapshotYear"] = 2025 - i
        r["MonthlyIncomeAtSnapshot"] = int(row["MonthlyIncome"] * (1 + random.uniform(-0.05, 0.15)))
        r["JobLevelAtSnapshot"] = min(5, row["JobLevel"] + (1 if random.random() < 0.15 else 0))
        rows_list.append(r)
    return pd.DataFrame(rows_list)

print("Generation des snapshots historiques...")
historical = pd.concat([add_history(row) for _, row in df.iterrows()], ignore_index=True)

# == SAUVEGARDE ==
historical.to_csv(OUTPUT_FILE, index=False)

print()
print("=" * 70)
print(f"Dataset SopraHR IT sauvegarde : {OUTPUT_FILE}")
print(f"Taille finale : {historical.shape}")
print()
depts = sorted(historical["Department"].unique())
print(f"Departements ({len(depts)}) : {depts}")
print()
roles = sorted(historical["JobRole"].unique())
print(f"JobRoles ({len(roles)}) : {roles}")
print()
ct = historical["ContractType"].value_counts()
cdi = ct.get("CDI", 0)
cdd = ct.get("CDD", 0)
total = len(historical)
print(f"CDI: {cdi} ({cdi/total*100:.0f}%) | CDD: {cdd} ({cdd/total*100:.0f}%)")
print()
print(f"Seniority : {historical['Seniority'].value_counts().to_dict()}")
print()
print(f"EducationField : {sorted(historical['EducationField'].unique())}")
print()
example = historical[["EmployeeNumber", "Department", "JobRole", "Seniority", "Skills", "Certifications"]].sample(3, random_state=42)
print("Exemples :")
print(example.to_string(index=False))
print("=" * 70)

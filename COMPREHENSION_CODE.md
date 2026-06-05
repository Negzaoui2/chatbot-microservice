# Documentation — Compréhension du Code

## 1. C'est quoi `self` ? (ex: `self._api_key`)

`self` est une référence à **l'objet lui-même** (l'instance de la classe). C'est l'équivalent de `this` en Java/JavaScript.

```python
class LlmService:
    def __init__(self, api_key):
        self._api_key = api_key  # stocke api_key DANS cet objet

service = LlmService(api_key="gsk_abc123")
# self = service
# self._api_key = "gsk_abc123"
```

Sans `self`, la variable disparaît après `__init__()`. Avec `self`, elle reste accessible dans **toutes les méthodes** de l'objet.

---

## 2. C'est quoi `RandomForestClassifier | None = None` ?

```python
self._model: RandomForestClassifier | None = None
```

| Partie | Signification |
|---|---|
| `self._model` | Attribut privé de l'objet |
| `: RandomForestClassifier \| None` | **Type annotation** — peut être un RandomForestClassifier OU None |
| `= None` | **Valeur initiale** — au départ le modèle n'existe pas |

Le `|` (pipe) = un **OU de types** (union). Le modèle est `None` au début, puis devient un vrai `RandomForestClassifier` après l'entraînement dans `_train()`.

---

## 3. Où le code accède au dataset ?

Le nom du fichier (`HR-Enriched-History-With-IT.csv`) est défini dans le fichier **`.env`** :

```
DATASET_PATH=data/HR-Enriched-History-With-IT.csv
```

Il est lu par `settings.py` (Pydantic) puis utilisé par 5 services via `pd.read_csv(settings.dataset_path)` :

| Service | Rôle |
|---|---|
| `rag_service.py` | Indexer les documents pour la recherche sémantique |
| `prediction_service.py` | Entraîner le modèle ML (Random Forest) |
| `analytics_service.py` | Calculer les statistiques/rapports |
| `planning_service.py` | Recommandations de staffing |
| `export_service.py` | Générer les PDF/Excel |

**Important** : Le LLM (LLaMA) n'accède jamais directement au dataset. C'est le RAG qui cherche les docs pertinents, puis les envoie au LLM comme contexte.

---

## 4. Sentence-transformers et LLaMA 3.3 — comment ils travaillent ensemble ?

Ce sont **deux modèles différents** qui travaillent **en séquence** (pas en parallèle) :

| Composant | Rôle | Étape |
|---|---|---|
| **sentence-transformers** | Transformer la question en vecteur et trouver les documents similaires | **Étape 1 : Retrieval (Recherche)** |
| **LLaMA 3.3 (via Groq)** | Prendre les documents trouvés + la question et formuler une réponse | **Étape 2 : Generation (Rédaction)** |

C'est le principe du **RAG** (Retrieval-Augmented Generation) :
1. Récupérer le contexte pertinent (sentence-transformers)
2. Générer une réponse basée sur ce contexte (LLaMA)

---

## 5. C'est quoi `DATASET_TEXT_COLUMNS` et `RAG_TOP_K` dans le `.env` ?

### `DATASET_TEXT_COLUMNS`
Ce ne sont **pas toutes les colonnes** du dataset (qui en a 44). Ce sont les **14 colonnes textuelles** choisies pour la recherche sémantique. Les colonnes numériques (Age, MonthlyIncome) ne servent pas à la recherche par texte.

> **Note** : dans le code actuel de `rag_service.py`, cette variable n'est pas utilisée — le code a sa propre liste `priority_cols` en dur.

### `RAG_TOP_K=5`
Quand l'utilisateur pose une question, le RAG retourne les **5 documents les plus pertinents** parmi les 5217 indexés. Ces 5 documents sont envoyés au LLM comme contexte pour formuler la réponse.

---

## 6. Architecture globale du chatbot

```
Utilisateur → Angular Frontend → POST /chat → FastAPI (main.py)
                                                    ↓
                                            Intent Detector
                                        (détecte le type de question)
                                                    ↓
                            ┌────────────────────────────────────────┐
                            │  question → RAG + LLM                  │
                            │  prediction → PredictionService (ML)   │
                            │  report → AnalyticsService             │
                            │  suggestion → PlanningService           │
                            │  export → ExportService (PDF/Excel)    │
                            └────────────────────────────────────────┘
```

---

## 7. Comment l'accuracy est calculée automatiquement ? (pas fixée à 99%)

L'accuracy n'est **PAS codée en dur**. Elle est **calculée automatiquement** par scikit-learn.

### Le processus (dans `prediction_service.py`)

```python
# 1. Diviser le dataset en 2 parties
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#         80% pour apprendre         20% pour tester

# 2. Le modèle apprend SEULEMENT sur les 80%
self._model.fit(X_train, y_train)

# 3. On demande au modèle de DEVINER sur les 20% qu'il n'a jamais vu
y_pred = self._model.predict(X_test)

# 4. On compare ses prédictions avec les VRAIES réponses
self._accuracy = accuracy_score(y_test, y_pred)
#                                ↑ vrai    ↑ prédit
```

### Exemple simplifié

| Employé | Vrai (y_test) | Prédit (y_pred) | Correct ? |
|---|---|---|---|
| #1 | Va partir | Va partir | ✅ |
| #2 | Reste | Reste | ✅ |
| #3 | Va partir | Reste | ❌ |

→ 2 corrects sur 3 = **66.7% accuracy**

Dans notre cas : sur ~1043 employés de test, le modèle devine correctement ~1034 → **99.1%**

### Conclusion
- On n'a jamais écrit `accuracy = 0.99` dans le code
- C'est `accuracy_score()` qui compare et calcule le résultat
- Si on change le dataset ou les features, l'accuracy changera automatiquement

pour tester et voir l'accuracy on tappe cette commande:
python -c "from app.services.prediction_service import PredictionService; p = PredictionService(); print(f'Accuracy: {p._accuracy:.2%}')"
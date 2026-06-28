# 🔮 Churn Prediction — Telco Customer Churn

Projet de classification binaire bout-en-bout conçu pour un portfolio Data Science.  
L'objectif est de prédire le risque de churn d'un client télécom à partir de ses caractéristiques contractuelles et d'usage, avec explainabilité des prédictions via SHAP.

**Demo live** → [huggingface.co/spaces/Naam27062000/churn-prediction](https://huggingface.co/spaces/Naam27062000/churn-prediction)

---

## Problème métier

Un client qui churne représente un coût élevé (acquisition d'un nouveau client coûte 5 à 7x plus cher que de retenir un client existant). L'enjeu est d'identifier les clients à risque **avant** qu'ils ne partent, pour déclencher une action de rétention ciblée.

---

## Dataset

- **Source** : [Telco Customer Churn — IBM Watson](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Taille** : 7 043 clients × 21 features
- **Cible** : `Churn` (Yes / No) — déséquilibre ~73% / 27%

---

## Pipeline

```
EDA statistique
    → Mann-Whitney U (variables continues)
    → Chi² d'indépendance (variables catégorielles)
    → Variables les plus discriminantes : Contract, OnlineSecurity, TechSupport

Preprocessing
    → Encodage binaire + One-Hot Encoding
    → Feature engineering : AvgMonthlyCharge, NbServices, TenureGroup
    → Train/Test split stratifié (80/20)
    → StandardScaler (dans le pipeline sklearn)

Modélisation — StratifiedKFold (5 folds), métrique : PR-AUC
    ┌─────────────────────┬────────┬────────┬───────┐
    │ Modèle              │ PR-AUC │ Recall │ F1    │
    ├─────────────────────┼────────┼────────┼───────┤
    │ LogisticRegression  │ 0.664  │ 0.797  │ 0.628 │
    │ RandomForest        │ 0.616  │ 0.647  │ 0.608 │
    │ XGBoost             │ 0.612  │ 0.601  │ 0.578 │
    └─────────────────────┴────────┴────────┴───────┘

Tuning — GridSearchCV (C, l1_ratio) + MLflow tracking
    → Best params : C=1, l1_ratio=1.0 (L1 pure)
    → PR-AUC test : 0.636 | Recall : 0.794 | Precision : 0.499

Explainability — SHAP LinearExplainer
    → Top features : tenure, Contract_Two year, InternetService_Fiber optic
```

---

## Résultats

| Métrique  | Score |
|-----------|-------|
| PR-AUC    | 0.636 |
| Recall    | 0.794 |
| Precision | 0.499 |
| F1        | 0.613 |

Le recall de 79% signifie que le modèle détecte 4 churners sur 5 — ce qui est l'objectif métier principal (minimiser les faux négatifs).

---

## Architecture de déploiement

```
FastAPI (/predict)
    ↑ requests
Streamlit (interface utilisateur)
    ↓
Hugging Face Spaces (Docker)
```

---

## Stack technique

| Catégorie | Outils |
|-----------|--------|
| Modélisation | scikit-learn, XGBoost |
| Explainability | SHAP |
| Tracking | MLflow |
| API | FastAPI, uvicorn |
| Interface | Streamlit |
| Déploiement | Hugging Face Spaces (Docker) |

---

## Structure du projet

```
churn-prediction/
│
├── app.py                        # Point d'entrée Hugging Face Spaces
├── Dockerfile                    # Image Docker unique (API + Streamlit)
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/
│
├── models/
│   └── churn_pipeline.pkl        # Pipeline sklearn sérialisé (preprocessor + modèle)
│
├── notebooks/
│   ├── 01_eda.ipynb              # EDA + modélisation + SHAP
│   ├── mlruns/                   # Runs MLflow
│   └── reports/
│       └── figures/
│           ├── shap_global.png
│           └── shap_beeswarm.png
│
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py               # FastAPI — endpoints /health et /predict
│   └── app/
│       └── streamlit_app.py      # Interface Streamlit
│
└── tests/
```

---

## Lancer en local

```bash
# Activer le venv
venv\Scripts\activate

# API
uvicorn src.api.main:app --reload

# Interface (second terminal)
streamlit run src/app/streamlit_app.py
```

---

## Choix méthodologiques clés

- **PR-AUC plutôt qu'accuracy** : avec 73% de non-churn, l'accuracy est une métrique trompeuse. La PR-AUC est plus informative sur les classes déséquilibrées.
- **class_weight="balanced"** : pénalise davantage les erreurs sur la classe minoritaire (churners) sans recourir à SMOTE.
- **Pipeline sklearn complet** : le preprocessor et le modèle sont sérialisés ensemble — l'API reçoit des données brutes et gère tout en interne.
- **SHAP LinearExplainer** : adapté à la régression logistique, permet une double lecture globale (summary plot) et individuelle (waterfall) pour rendre le modèle explicable.

---

## Auteur

**Nasseme** — Master 2 Mathématiques, Applications et Data Science (Sorbonne Université)

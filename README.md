# Prédiction des Incendies de Forêts au Maroc (DDDM)

Ce projet a été réalisé dans le cadre du module **Data-Driven Decision Making (DDDM)**. Il met en œuvre un système complet d'aide à la décision (DSS) basé sur la donnée pour prédire les risques de départs de feux de forêts au Maroc (2010–2022) et optimiser la planification de prévention opérationnelle et stratégique.

---

## 🎯 Problématique Métier & KPIs

### 1. Question Décisionnelle Centrale
**« Comment prédire le risque d'incendies de forêts par zone géographique et conditions météorologiques au Maroc afin d'optimiser l'intervention de la Protection Civile et l'efficacité des campagnes de sensibilisation ? »**

### 2. L'Arbre des Indicateurs (KPI Tree)
Le KPI Tree relie les objectifs de préservation environnementale et de réduction de budget d'urgence de la Protection Civile aux indicateurs opérationnels et modèles prédictifs.

```text
Objectif Stratégique (Ministère de l'Intérieur & Eaux et Forêts)
├── Budget National de Lutte contre le Feu (Métrique Stratégique)
│   ├── Coût Logistique par Intervention (Métrique Tactique)
│   │   └── Distance Moyenne Station-Feu (Métrique Opérationnelle)
│   └── Nombre d'Interventions Lourdes (Métrique Tactique)
│       ├── Nombre d'Incendies Évités (Métrique Opérationnelle)
│       └── Taux de Faux Négatifs (Recall Modèle) (Métrique Opérationnelle)
└── Préservation du Couvert Forestier (Hectares) (Métrique Stratégique)
    ├── Temps de Réponse des Secours (Métrique Tactique)
    │   ├── Probabilité d'Alerte Prédite (XGBoost) (Métrique Opérationnelle)
    │   └── Indice de Danger Météo Moyen (FWI) (Métrique Opérationnelle)
    └── Taux d'Adoption des Comportements Prudents (Métrique Tactique)
        ├── Taux d'Impact des Alertes SMS (A/B Test) (Métrique Opérationnelle)
        └── Densité Végétale Résiduelle (NDVI) (Métrique Opérationnelle)
```

### 3. Business Case & ROI de la Démarche Data
Prévenir un feu via des alertes géociblées est **très économique** comparé au coût de lutte active :
* **Coût d'intervention moyen pour 1 incendie lourd** : 150 000 MAD (personnel, hélicoptères bombardiers d'eau, logistique).
* **Coût d'une alerte SMS ciblée** : 0.18 MAD / personne.

#### Simulation de ROI (Modèle XGBoost) :
* Pour 12 communes ciblées représentant 96 000 SMS : **Coût de campagne** = $96\ 000 \times 0.18\text{ MAD} = 17\ 280\text{ MAD}$.
* Si la campagne permet d'éviter seulement **2 incendies majeurs** (grâce à la prudence accrue des populations locales) : **Coûts évités** = $2 \times 150\ 000\text{ MAD} = 300\ 000\text{ MAD}$.
* **Bénéfice Net** = $300\ 000\text{ MAD} - 17\ 280\text{ MAD} = 282\ 720\text{ MAD}$.
* **Retour sur Investissement (ROI)** : **$1\ 636\%$** (chaque dirham investi rapporte 16,36 MAD de coûts d'intervention évités).

---

## 🏗️ Architecture du Système Décisionnel

```text
+-----------------------+      +-----------------------+
|    NASA Satellites    |      |       NOAA GSOD       |
|    (NASA FIRMS)       |      |     (Météo Lags)      |
+-----------+-----------+      +-----------+-----------+
            |                              |
            +--------------+---------------+
                           |
                           v
               +-------------------------+
               | Ingestion & Audit Data  | <--- data_loader.py
               +------------+------------+
                            |
                            v
               +-------------------------+
               |   FWI & Lags Temporel   | <--- preprocessor.py
               +------------+------------+
                            |
            +--------------+--------------+
            |                             |
            v                             v
+-----------------------+     +-----------------------+
|  Clustering Risques   |     |  Modèles Incendies    | <--- models.py
|  (K-Means k=3 Zones)  |     |  (XGBoost, RF, LR)    |
+-----------+-----------+     +-----------+-----------+
            |                             |
            |                             v
            |                 +-----------------------+
            |                 |  Interprétabilité     | <--- SHAP Explainer
            |                 |  (Global & Local)     |
            |                 +-----------+-----------+
            |                             |
            +--------------+--------------+
                           |
                           v
               +-------------------------+
               |  Interactive Dashboard  | <--- app.py (Streamlit)
               | (Direction, Opérations, |
               |   Prévention, A/B Test) |
               +-------------------------+
```

---

## 📂 Structure du Répertoire
```text
DDDM-Projet/
├── data/
│   ├── Morocco_Wildfire_Predictions.csv  # Dataset filtré à 100 000 lignes
│   └── model_artifacts.pkl              # Objets de sauvegarde (Modèle XGBoost, scaler, clusters)
├── src/
│   ├── data_loader.py         # Fonctions de chargement et d'audit qualité des données
│   ├── preprocessor.py        # Nettoyage, conversions et feature engineering (FWI, régions)
│   ├── models.py              # Entraînement, évaluation et explicabilité SHAP
│   ├── train_models.py        # Script d'exécution global de la pipeline ML
│   └── create_notebook.py     # Générateur du Jupyter notebook d'analyse
├── notebooks/
│   └── eda_and_modeling.ipynb # Analyse exploratoire, tests statistiques et modélisation ML
├── dashboard/
│   └── app.py                 # Application Streamlit décisionnelle (5 vues interactives)
├── docs/
│   └── data_story.pptx        # Structure de la présentation
│   └── ab_test_plan.pdf       # Protocole expérimental de test A/B
├── requirements.txt           # Dépendances Python
├── run.bat                    # Script de démarrage automatique pour Windows
└── README.md                  # Ce guide explicatif
```

---

## 🛠️ Instructions d'Installation & Lancement

Le projet est conçu pour être lancé de manière extrêmement simple sur Windows grâce au script de démarrage automatique `run.bat`.

### Option A : Démarrage Automatique (Recommandé)
Double-cliquez simplement sur le fichier `run.bat` à la racine du projet. 
Ce script va automatiquement :
1. Installer les dépendances nécessaires.
2. Lancer la pipeline d'entraînement des modèles de classification.
3. Lancer le tableau de bord Streamlit localement dans votre navigateur.

---

### Option B : Lancement Manuel par Étape
Si vous préférez exécuter les scripts étape par étape dans votre terminal :

**1. Installation des dépendances :**
```bash
pip install -r requirements.txt
```

**2. Entraînement de la pipeline Machine Learning :**
```bash
python src/train_models.py
```

**3. Démarrage du dashboard interactif Streamlit :**
```bash
streamlit run dashboard/app.py
```

---

## 📈 Résultats des Tests Statistiques (EDA)
*   **Température vs Incendies** : Le test **t de Student** démontre de manière hautement significative ($p < 0.001$) que la température moyenne est nettement supérieure lors des jours de départs de feux.
*   **Régions vs Incendies** : Le test du **Chi-deux** valide de façon statistiquement significative ($p < 0.05$) l'association entre la région géographique (Nord, Centre, Sud) et l'occurrence des feux.

---

## 🏆 Performances Prédictives Clés (XGBoost)
*   **Exactitude (Accuracy)** : **93.18%**
*   **Rappel (Recall)** : **94.51%** (Crucial pour minimiser les départs de feux manqués).
*   **Score F1** : **93.25%**
*   **ROC-AUC** : **98.22%** (Capacité exceptionnelle à différencier les situations à haut risque des situations sûres).

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
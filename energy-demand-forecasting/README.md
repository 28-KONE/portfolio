# Prévision de consommation énergétique

## Enjeu métier
Dans les secteurs de l’énergie et de l’industrie, une mauvaise anticipation de la consommation peut entraîner :
- des surcoûts de production,
- des déséquilibres entre offre et demande,
- des risques opérationnels importants.

L’objectif de ce projet est de **prédire la consommation énergétique à court et moyen terme** afin d’aider à la planification et à l’optimisation des ressources.

---

## Objectifs du projet
- Analyser des séries temporelles de consommation énergétique
- Intégrer des **variables exogènes** (météo, calendrier)
- Construire des modèles prédictifs robustes et interprétables
- Comparer plusieurs approches de modélisation

---

## Démarche suivie

### Analyse exploratoire
- Visualisation des tendances, saisonnalités et cycles
- Analyse des effets calendaires (jours ouvrés, week-ends, saisons)
- Détection de valeurs aberrantes et ruptures de régime

### Feature engineering temporel
- Création de variables de retard (*lags*)
- Statistiques glissantes (*rolling mean, rolling std*)
- Encodage des informations calendaires
- Intégration des données météo comme variables explicatives

### Modélisation
- Modèles de régression classiques comme baseline
- Modèles de Machine Learning :
  - Random Forest
  - Gradient Boosting / XGBoost
- Comparaison des performances selon l’horizon de prévision

### Évaluation
- Métriques adaptées aux séries temporelles (MAE, RMSE, MAPE)
- Validation temporelle (respect de l’ordre chronologique)
- Analyse des erreurs sur les pics de consommation

---

## Ce projet m’a permis de faire :
- Manipuler des **séries temporelles réelles**
- Concevoir des features pertinentes pour des données séquentielles
- Comprendre les limites des modèles selon l’horizon de prévision
- Travailler sur une problématique directement applicable au secteur de l’énergie

---

## Contenu du repository
- `notebooks/` : exploration, feature engineering et modélisation
- `data/` : jeux de données (consommation, météo)
- `README.md` : description du projet et des choix méthodologiques

# Credit Scoring

**Enjeu métier** :  
Le crédit scoring est un problème central pour les institutions financières.  
Une mauvaise prédiction peut entraîner soit :
- une perte financière (accorder un crédit à un client à risque),
- soit une perte d’opportunité (refuser un client solvable).

L’objectif n’est donc pas uniquement de maximiser l’accuracy, mais de **trouver un compromis entre performance, interprétabilité et gestion du risque**, avec une attention particulière portée au **recall de la classe “défaut”**.

**Démarche suivie** :
1. Compréhension du problème et des données
2. Mise en place d’une baseline simple
3. Amélioration progressive via :
   - feature engineering
   - normalisation / réduction de dimension
   - sélection de variables
4. Comparaison rigoureuse de plusieurs algorithmes
5. Optimisation et automatisation dans un pipeline
6. Préparation à une mise en production (scoring)

**Ce projet m'a appris à :**
- Structurer un projet de Machine Learning de bout en bout
- Comparer objectivement plusieurs modèles via validation croisée
- Adapter les métriques d’évaluation au contexte métier
- Réduire la complexité d’un modèle sans sacrifier la performance
- Industrialiser une solution via des pipelines reproductibles

**Notebooks disponibles** :

- **Projet_AS.ipynb**
- **Projet_AS_Partie2.ipynb**


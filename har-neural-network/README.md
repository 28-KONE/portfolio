# Human Activity Recognition (HAR)  
## Classification d’activités humaines à partir de données de capteurs

## Contexte
Les systèmes de reconnaissance d’activités humaines sont au cœur de nombreuses applications :
- santé connectée (suivi d’activité, prévention),
- transports intelligents,
- industrie (ergonomie, sécurité, maintenance humaine).

Ce projet vise à **classifier des activités humaines** à partir de données issues de capteurs inertiels embarqués (accéléromètre, gyroscope).

---

## Objectifs du projet
- Exploiter des données de capteurs multivariées et bruitées
- Comparer différentes architectures de réseaux de neurones
- Évaluer la capacité de généralisation des modèles
- Analyser les représentations apprises par les réseaux

---

## Données
Les données utilisées dans ce projet ne sont pas incluses dans le repository en raison de leur taille.
- Données issues de smartphones portés par des individus
- Capteurs : accéléromètre et gyroscope
- Activités : marche, montée/descente d’escaliers, position statique, etc.

### Source
- Dataset : *Human Activity Recognition Using Smartphones*
- Source officielle : UCI Machine Learning Repository

### Instructions
1. Télécharger le dataset depuis la source officielle
2. Extraire les fichiers dans le dossier `data/`
3. Exécuter les notebooks dans l’ordre indiqué

---

## Approche méthodologique

### Prétraitement
- Normalisation des signaux
- Segmentation temporelle
- Organisation des données par fenêtres temporelles

### Modélisation Deep Learning
- Réseau **MLP** comme baseline
- Réseaux **CNN** pour capturer les motifs temporels
- Comparaison des architectures et hyperparamètres

### Évaluation
- Accuracy, précision, recall, F1-score
- Matrices de confusion
- Analyse du sur-apprentissage et de la régularisation

### Analyse des représentations
- Visualisation des embeddings latents (t-SNE / UMAP)
- Étude de la séparabilité des classes dans l’espace appris

---

## Résultats
- Les CNN montrent une meilleure capacité à capter la dynamique temporelle
- Bonne généralisation sur les activités dynamiques
- Limites observées sur les activités statiques proches

---

## Ce projet m’a permis de :
- Appliquer le Deep Learning à des **données capteurs réelles**
- Comprendre l’impact de l’architecture sur la performance
- Analyser les représentations internes des réseaux
- Relier des choix techniques à des enjeux concrets de santé et d’industrie

---

## Contenu du repository
- `notebooks/` : preprocessing, entraînement et évaluation
- `data/` : données capteurs
- `README.md` : description et analyse du projet

---

## Rapport détaillé

Un rapport complet accompagne ce projet et présente de manière approfondie :
- l’analyse exploratoire des données,
- la comparaison systématique des architectures (MLP profonds),
- l’étude du sur-apprentissage et des techniques de régularisation,
- l’évaluation finale sur jeu de test,
- l’interprétabilité des modèles via t-SNE et UMAP.

**Rapport** : `report/HAR_Deep_Learning_Report_Damba_KONE.pdf`

> Ce document met l’accent sur la justification des choix méthodologiques et l’analyse critique des résultats, dans le cadre d'une introduction à la recherche appliquée.

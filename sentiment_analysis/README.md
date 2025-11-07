# 🎬 Analyse de Sentiment - NLP & Transformers

## 🧠 Objectif
Ce projet vise à construire un modèle capable de déterminer si une critique de film est **positive** ou **négative**.  
Il illustre la transition entre les **représentations classiques du langage (Word2Vec)** et les **modèles modernes basés sur les Transformers (DistilBERT)**.

---

## 🪜 Étapes du projet
1. **Prétraitement & exploration** des données (IMDB dataset)
2. **Entraînement Word2Vec** + classification traditionnelle
3. **Fine-tuning DistilBERT** sur les mêmes données
4. **Comparaison des performances**
5. **Interface Streamlit** pour tester le modèle

---

## 📂 Structure

sentiment_analysis/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── word2vec/
│   │   └── word2vec.model
│   ├── classifiers/
│   │   └── logistic_word2vec.joblib
│   └── distilbert_finetuned/
├── scripts/
│   ├── download_data.py
│   ├── preprocess.py
│   ├── train_word2vec.py
│   ├── train_classifier.py
│   ├── finetune_transformer.py
│   └── evaluate.py
├── streamlit_app.py
├── utils.py
├── requirements.txt
└── README.md


---

## Installation
1. Crée un virtualenv / conda env
2. `pip install -r requirements.txt`
3. `python -m spacy download en_core_web_sm`

   
## ⚙️ Lancer l'application streamlit

1. `python scripts/download_data.py`
2. `python scripts/preprocess.py`
3. `python scripts/train_word2vec.py`
4. `python scripts/train_classifier.py`
5. `python scripts/finetune_transformer.py`
6. `python scripts/evaluate.py`
7. `streamlit run streamlit_app.py`


## 🧩 Technologies

- Python, Pandas, NumPy, Scikit-learn
- NLTK / spaCy / Gensim
- Transformers (Hugging Face), PyTorch
- Streamlit pour le déploiement

## ✨ Résultat attendu

Une application simple et interactive :

Entrée texte → modèle DistilBERT → résultat de sentiment

Comparaison entre Word2Vec et Transformers 



## Notes
- Application déjà deployé avec Streamlit : [https://28-kone-portfolio-sentiment-analysisstreamlit-app-v1nprb.streamlit.app/]



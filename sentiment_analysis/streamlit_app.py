# streamlit_app.py
import streamlit as st
from pathlib import Path
import joblib
from gensim.models import Word2Vec
from transformers import pipeline
import numpy as np
from utils import get_device


st.set_page_config(page_title="CinéMind — Analyse d'opinions", layout="centered")
st.title("🎬 CinéMind — AI for Movie Review Analysis")
st.write("Compare les performances entre Word2Vec et DistilBERT sur des critiques de films.")


# Chargement des modèles

@st.cache_resource
def load_word2vec_and_clf():
    w2v_path = Path("models/word2vec/word2vec.model")
    clf_path = Path("models/classifiers/logistic_word2vec.joblib")
    w2v = clf = None
    if w2v_path.exists() and clf_path.exists():
        w2v = Word2Vec.load(str(w2v_path))
        clf = joblib.load(str(clf_path))
    return w2v, clf

@st.cache_resource
def load_transformer():
    model_dir = Path("models/distilbert_finetuned_final")
    if model_dir.exists():
        device = 0 if get_device() == "cuda" else -1
        cl = pipeline("text-classification", model=str(model_dir), device=device, truncation=True)
        return cl
    return None

w2v, clf = load_word2vec_and_clf()
distil_pipeline = load_transformer()


# Fonctions de prédiction

def word2vec_predict(text):
    toks = text.split()
    if w2v is None or clf is None:
        return None
    vecs = [w2v.wv[t] for t in toks if t in w2v.wv]
    vec = np.mean(vecs, axis=0) if vecs else np.zeros(w2v.vector_size, dtype=float)
    label = clf.predict([vec])[0]
    probs = clf.predict_proba([vec])[0]
    return "POSITIF" if label == 1 else "NEGATIF", probs.max()

def distil_predict(text):
    if distil_pipeline is None:
        return None
    out = distil_pipeline(text[:512])  
    lab = out[0]["label"].upper()
    score = out[0]["score"]
    if lab.startswith("LABEL_"):
        lab = "POSITIF" if lab.endswith("1") else "NEGATIF"
    return lab, score


# Interface Streamlit

user_input = st.text_area("🎥 Entrez une critique de film :", height=200)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Word2Vec + Logistic")
    if st.button("Analyser (Word2Vec)"):
        if not user_input.strip():
            st.warning("⚠️ Entrez du texte.")
        else:
            r = word2vec_predict(user_input.lower())
            if r is None:
                st.info("ℹ️ Modèles Word2Vec / classifieur non trouvés.")
            else:
                label, score = r
                st.success(f"**{label}** (confiance {score:.2f})")

with col2:
    st.subheader("DistilBERT finetuné")
    if st.button("Analyser (DistilBERT)"):
        if not user_input.strip():
            st.warning("⚠️ Entrez du texte.")
        else:
            r = distil_predict(user_input)
            if r is None:
                st.info("ℹ️ Modèle DistilBERT non trouvé.")
            else:
                label, score = r
                st.success(f"**{label}** (confiance {score:.2f})")


# Nouveau bouton de comparaison

st.markdown("---")
if st.button("🔍 Comparer les deux modèles"):
    if not user_input.strip():
        st.warning("⚠️ Entrez du texte pour comparer.")
    else:
        r1 = word2vec_predict(user_input.lower())
        r2 = distil_predict(user_input)
        if r1 is None or r2 is None:
            st.info("ℹ️ Un des modèles n’est pas disponible.")
        else:
            l1, s1 = r1
            l2, s2 = r2
            st.subheader("Résultats comparés :")
            c1, c2 = st.columns(2)
            with c1:
                st.write("🧠 **Word2Vec + Logistic**")
                st.write(f"Résultat : {l1}")
                st.write(f"Confiance : {s1:.2f}")
            with c2:
                st.write("🤖 **DistilBERT finetuné**")
                st.write(f"Résultat : {l2}")
                st.write(f"Confiance : {s2:.2f}")

st.markdown("---")
st.caption("⚙️ Ce modèle DistilBERT est entraîné sur des critiques de films en anglais.")

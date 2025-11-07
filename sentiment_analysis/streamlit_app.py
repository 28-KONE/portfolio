# 🎬 CineMind — Movie Review Sentiment Analysis
# Interactive & Visual Comparison: Word2Vec + DistilBERT

import streamlit as st
from pathlib import Path
import joblib
from gensim.models import Word2Vec
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from utils import get_device
from huggingface_hub import hf_hub_download


# STREAMLIT CONFIG

st.set_page_config(
    page_title="CineMind — IA for Movie Review Analysis",
    page_icon="🎬",
    layout="centered"
)

st.markdown(
    """
    <h1 style="text-align:center;">🎬 <b>CineMind</b></h1>
    <p style="text-align:center;">
        Discover if a movie review is <b>positive</b> or <b>negative</b> according to two AIs:
        <br>Word2Vec + Logistic Regression 🧠 and DistilBERT 🤖
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")


# PATHS AND MODEL NAMES

DISTILBERT_MODEL = "28-KONE/sentiment-analysis-distilbert"
WORD2VEC_MODEL_PATH = "models/word2vec/word2vec.model"
CLASSIFIER_PATH = "models/classifiers/logistic_word2vec.joblib"


# LOAD MODELS

@st.cache_resource
def load_word2vec_and_clf():
    w2v_repo = "28-KONE/sentiment-analysis-word2vec"
    w2v_path = hf_hub_download(repo_id=w2v_repo, filename="word2vec.model")
    clf_path = hf_hub_download(repo_id=w2v_repo, filename="logistic_word2vec.joblib")
    w2v = Word2Vec.load(w2v_path)
    clf = joblib.load(clf_path)
    return w2v, clf


@st.cache_resource
def load_transformer():
    device = 0 if get_device() == "cuda" else -1
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_MODEL)
    clf_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    return clf_pipeline

w2v, clf = load_word2vec_and_clf()
distil_pipeline = load_transformer()


# PREDICTION FUNCTIONS

def word2vec_predict(text):
    if w2v is None or clf is None:
        return None
    toks = text.split()
    vecs = [w2v.wv[t] for t in toks if t in w2v.wv]
    vec = np.mean(vecs, axis=0) if vecs else np.zeros(w2v.vector_size, dtype=float)
    label = clf.predict([vec])[0]
    probs = clf.predict_proba([vec])[0]
    return ("POSITIVE" if label == 1 else "NEGATIVE"), float(probs.max())

def distil_predict(text):
    if distil_pipeline is None:
        return None
    out = distil_pipeline(text[:512])
    lab = out[0]["label"].upper()
    score = float(out[0]["score"])
    if lab.startswith("LABEL_"):
        lab = "POSITIVE" if lab.endswith("1") else "NEGATIVE"
    return lab, score


# MAIN INTERFACE

st.subheader("📝 Enter a movie review to analyze")
st.caption("Try your own text or use one of the sample reviews below.")

col_ex1, col_ex2 = st.columns(2)
with col_ex1:
    if st.button("🌟 Positive Example"):
        example = "This movie was amazing!"
        st.session_state["example_text"] = example
with col_ex2:
    if st.button("💔 Negative Example"):
        example = "This movie was awful. The plot made no sense and the acting was bad."
        st.session_state["example_text"] = example

user_input = st.text_area(
    "🎥 Your review:",
    value=st.session_state.get("example_text", ""),
    height=180
)


# MODEL RESULTS DISPLAY

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🧠 Word2Vec + Logistic Regression")
    if st.button("Analyze with Word2Vec"):
        if not user_input.strip():
            st.warning("⚠️ Please enter a review before analyzing.")
        else:
            r = word2vec_predict(user_input.lower())
            if r is None:
                st.info("ℹ️ Word2Vec or classifier model not available.")
            else:
                label, score = r
                emoji = "😁" if label == "POSITIVE" else "😞"
                color = "#90EE90" if label == "POSITIVE" else "#FFB6C1"
                st.markdown(
                    f"""
                    <div style="background-color:{color};padding:15px;border-radius:10px;text-align:center;">
                        <h3>{emoji} {label}</h3>
                        <p>Confidence: <b>{score:.2f}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

with col2:
    st.markdown("### 🤖 Fine-tuned DistilBERT")
    if st.button("Analyze with DistilBERT"):
        if not user_input.strip():
            st.warning("⚠️ Please enter a review before analyzing.")
        else:
            r = distil_predict(user_input)
            if r is None:
                st.info("ℹ️ DistilBERT model not available.")
            else:
                label, score = r
                emoji = "😁" if label == "POSITIVE" else "😞"
                color = "#ADD8E6" if label == "POSITIVE" else "#FFA07A"
                st.markdown(
                    f"""
                    <div style="background-color:{color};padding:15px;border-radius:10px;text-align:center;">
                        <h3>{emoji} {label}</h3>
                        <p>Confidence: <b>{score:.2f}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# MODEL COMPARISON

st.markdown("---")
st.markdown("## Compare Both Models")

if st.button("Compare the Two AIs"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a review to compare.")
    else:
        r1 = word2vec_predict(user_input.lower())
        r2 = distil_predict(user_input)
        if r1 is None or r2 is None:
            st.info("ℹ️ One of the models is not available.")
        else:
            l1, s1 = r1
            l2, s2 = r2

            colc1, colc2 = st.columns(2)
            with colc1:
                st.markdown(f"### 🧠 Word2Vec + Logistic Regression")
                st.progress(float(s1))
                st.write(f"**Result:** {l1}")
                st.write(f"**Confidence:** {s1:.2f}")

            with colc2:
                st.markdown(f"### 🤖 Fine-tuned DistilBERT")
                st.progress(float(s2))
                st.write(f"**Result:** {l2}")
                st.write(f"**Confidence:** {s2:.2f}")

# ABOUT SECTION

st.markdown("---")
with st.expander("ℹ️ About the Models"):
    st.write("""
    **🧠 Word2Vec + Logistic Regression**
    - A classic model using numeric representations of words.
    - Captures basic word associations to predict sentiment.
    - Fast and lightweight, but less accurate.

    **🤖 Fine-tuned DistilBERT**
    - A modern AI model derived from BERT, trained on millions of text samples.
    - Understands sentence context and linguistic nuances.
    - More accurate for subtle expressions of sentiment.
    """)

st.caption("⚙️ Fine-tuned DistilBERT model by 28-KONE • CineMind Educational Project")


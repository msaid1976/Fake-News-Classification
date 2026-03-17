import base64
import ast
import json
import os
import re
import pickle
import string
from io import StringIO
from pathlib import Path
from urllib.parse import urljoin, urlparse

import emoji
import joblib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from bs4 import BeautifulSoup
from matplotlib.colors import LinearSegmentedColormap
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split

try:
    import trafilatura
except Exception:
    trafilatura = None


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "Models"
PIPELINE_FILE_ORDER = [
    "best_model_pipeline.pkl",
    "linear_svm_pipeline.pkl",
    "logistic_regression_pipeline.pkl",
    "naive_bayes_pipeline.pkl",
    "random_forest_pipeline.pkl",
]
RESULT_FILE_ORDER = [
    "benchmark_results.pkl",
    "model_metrics.pkl",
    "model_RF_results.pkl",
    "model_LSVM_results.pkl",
]

PRIMARY_COLORS = ["#1f6feb", "#2f9e44"]
FAKE_COLOR = "#d94841"
REAL_COLOR = "#3b874d"
NOTEBOOK_PRIMARY_COLORS = ["#3CB371", "#0f6096"]
BENCHMARK_FAKE_COLOR, BENCHMARK_REAL_COLOR = NOTEBOOK_PRIMARY_COLORS
LABEL_MAP = {0: "Fake News", 1: "Real News"}
DEFAULT_THRESHOLD = 0.50
DEFAULT_MIN_CHARS = 50
DEFAULT_MAX_CHARS = 1000
REAL_SOURCE_SAMPLE_LIMIT = 5
FAKE_PREDICT_SAMPLE_LIMIT = 5
NOTEBOOK_BENCHMARK_CELLS = {
    "baseline_comparison": 76,
    "robust_cv": 78,
    "tuned_comparison": 88,
    "evaluation": 91,
    "stage_comparison": 93,
}
MODEL_SLUG_TO_PIPELINE_FILE = {
    "linear_svm": "linear_svm_pipeline.pkl",
    "logistic_regression": "logistic_regression_pipeline.pkl",
    "naive_bayes": "naive_bayes_pipeline.pkl",
    "random_forest": "random_forest_pipeline.pkl",
}

NEWS_SOURCE_CATALOG = {
    "middle_east": {
        "label": "Middle East News",
        "homepage": "https://www.aljazeera.com/middle-east/",
        "feeds": [
            "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml",
            "https://www.aljazeera.com/xml/rss/all.xml",
        ],
        "fake_keywords": [
            "middle east",
            "gaza",
            "israel",
            "iran",
            "saudi",
            "qatar",
            "syria",
            "yemen",
            "lebanon",
            "palestine",
        ],
    },
    "usa": {
        "label": "USA News",
        "homepage": "https://www.npr.org/sections/news/",
        "feeds": [
            "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml",
            "https://feeds.npr.org/1001/rss.xml",
        ],
        "fake_keywords": [
            "u.s.",
            "united states",
            "washington",
            "america",
            "american",
            "trump",
            "biden",
            "congress",
            "senate",
        ],
    },
    "iran": {
        "label": "Iran News",
        "homepage": "https://www.tehrantimes.com/",
        "feeds": [
            "https://www.tehrantimes.com/rss",
            "https://en.irna.ir/rss",
        ],
        "fake_keywords": [
            "iran",
            "iranian",
            "tehran",
            "ayatollah",
            "islamic republic",
        ],
    },
    "malaysia": {
        "label": "Malaysia News",
        "homepage": "https://www.malaymail.com/news/malaysia",
        "feeds": [
            "https://www.malaymail.com/feed/rss/malaysia",
            "https://rss.thestar.com.my/rss/news/nation/",
        ],
        "fake_keywords": [
            "malaysia",
            "malaysian",
            "kuala lumpur",
            "putrajaya",
            "anwar",
            "selangor",
        ],
    },
    "asia": {
        "label": "Asia News",
        "homepage": "https://www.bbc.com/news/world/asia",
        "feeds": [
            "https://feeds.bbci.co.uk/news/world/asia/rss.xml",
            "https://feeds.npr.org/1004/rss.xml",
        ],
        "fake_keywords": [
            "asia",
            "china",
            "japan",
            "india",
            "singapore",
            "korea",
            "bangkok",
            "manila",
        ],
    },
    "europe": {
        "label": "Europe News",
        "homepage": "https://www.bbc.com/news/world/europe",
        "feeds": [
            "https://feeds.bbci.co.uk/news/world/europe/rss.xml",
            "https://feeds.npr.org/1004/rss.xml",
        ],
        "fake_keywords": [
            "europe",
            "eu",
            "ukraine",
            "britain",
            "france",
            "germany",
            "italy",
        ],
    },
    "africa": {
        "label": "Africa News",
        "homepage": "https://www.bbc.com/news/world/africa",
        "feeds": [
            "https://feeds.bbci.co.uk/news/world/africa/rss.xml",
            "https://feeds.npr.org/1004/rss.xml",
        ],
        "fake_keywords": [
            "africa",
            "sudan",
            "nigeria",
            "kenya",
            "ethiopia",
            "uganda",
            "ghana",
        ],
    },
    "global": {
        "label": "Global News",
        "homepage": "https://www.bbc.com/news/world",
        "feeds": [
            "https://feeds.bbci.co.uk/news/world/rss.xml",
            "https://feeds.npr.org/1004/rss.xml",
            "https://www.aljazeera.com/xml/rss/all.xml",
        ],
        "fake_keywords": [
            "world",
            "global",
            "international",
            "diplomacy",
            "summit",
            "conflict",
        ],
    },
}


def ensure_nltk_resource(resource_path: str, download_name: str) -> None:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name, quiet=True)


for resource_path, download_name in [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
]:
    ensure_nltk_resource(resource_path, download_name)


st.set_page_config(
    page_title="News Classification Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Source+Sans+3:wght@400;500;600;700&display=swap');

    :root {
        --ink: #102028;
        --sea: #335b84;
        --mint: #3b874d;
        --danger: #d94841;
        --paper: #f7fbfd;
        --card: #ffffff;
        --line: rgba(16, 32, 40, 0.10);
    }

    .stApp {
        background:
            radial-gradient(circle at top right, rgba(60, 179, 113, 0.14), transparent 32%),
            radial-gradient(circle at bottom left, rgba(15, 96, 150, 0.12), transparent 28%),
            #f5f9fb;
        color: var(--ink);
        font-family: 'Source Sans 3', sans-serif;
    }

    .hero-title {
        font-family: 'Fraunces', serif;
        font-size: 2.6rem;
        line-height: 1.05;
        color: #163243;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        font-size: 1.02rem;
        color: rgba(16, 32, 40, 0.82);
        margin-bottom: 1.15rem;
        max-width: 64rem;
    }

    .info-panel {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid var(--line);
        border-left: 6px solid var(--sea);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(16, 32, 40, 0.06);
    }

    .result-card {
        background: var(--card);
        border: 1px solid var(--line);
        border-left: 8px solid var(--mint);
        border-radius: 16px;
        padding: 1rem 1.15rem;
        box-shadow: 0 10px 28px rgba(16, 32, 40, 0.08);
        margin-bottom: 1rem;
    }

    .result-card.fake {
        border-left-color: var(--danger);
        background: #ffffff;
    }

    .result-card.real {
        border-left-color: var(--mint);
        background: #ffffff;
    }

    .result-label {
        font-family: 'Fraunces', serif;
        font-size: 1.5rem;
        color: #163243;
    }

    .result-meta {
        font-size: 0.98rem;
        margin-top: 0.35rem;
        color: rgba(16, 32, 40, 0.78);
    }

    .confidence-track {
        margin-top: 0.75rem;
        background: #dfe9ef;
        height: 12px;
        border-radius: 999px;
        overflow: visible;
        position: relative;
    }

    .confidence-fill {
        height: 12px;
        border-radius: 999px;
        position: relative;
        min-width: 8px;
    }

    .confidence-value {
        position: absolute;
        right: 0;
        top: 50%;
        transform: translate(50%, -50%);
        background: #ffffff;
        color: #163243;
        border-radius: 999px;
        padding: 0.05rem 0.38rem;
        font-size: 0.72rem;
        font-weight: 700;
        white-space: nowrap;
        box-shadow: 0 4px 10px rgba(16, 32, 40, 0.10);
    }

    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #335b84, #416f9c);
        color: #ffffff;
        border: 1px solid rgba(51, 91, 132, 0.42);
        border-radius: 12px;
        font-weight: 700;
        box-shadow: 0 8px 18px rgba(51, 91, 132, 0.24);
    }

    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg, #29496a, #335b84);
        color: #ffffff;
        border-color: rgba(51, 91, 132, 0.56);
    }

    div[data-testid="stButton"] > button p,
    div[data-testid="stButton"] > button span {
        color: #ffffff !important;
    }

    button[data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.88);
        color: #163243;
        border: 1px solid rgba(51, 91, 132, 0.18);
        border-radius: 12px 12px 0 0;
        font-weight: 400;
        padding: 0.55rem 1rem;
        margin-right: 0.2rem;
    }

    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] p {
        color: #163243;
        font-weight: 400 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #29496a, #335b84) !important;
        color: #ffffff !important;
        border-color: rgba(41, 73, 106, 0.72) !important;
        box-shadow: 0 8px 18px rgba(51, 91, 132, 0.24);
    }

    button[data-baseweb="tab"][aria-selected="true"] > div[data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
        font-weight: 400 !important;
    }

    .pipeline-card {
        background: rgba(255, 255, 255, 0.90);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1rem;
        min-height: 100%;
        box-shadow: 0 8px 22px rgba(16, 32, 40, 0.06);
    }

    .pipeline-step {
        font-size: 0.9rem;
        color: rgba(16, 32, 40, 0.86);
        margin-bottom: 0.42rem;
    }

    div[data-testid="stExpander"] details {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(51, 91, 132, 0.20);
        background: rgba(255, 255, 255, 0.82);
        box-shadow: 0 8px 24px rgba(16, 32, 40, 0.06);
    }

    div[data-testid="stExpander"] details > summary {
        background: linear-gradient(135deg, #335b84, #416f9c);
        color: #ffffff !important;
        border: none !important;
        padding: 0.85rem 1rem !important;
    }

    div[data-testid="stExpander"] details > summary:hover {
        background: linear-gradient(135deg, #29496a, #335b84);
    }

    div[data-testid="stExpander"] details > summary span,
    div[data-testid="stExpander"] details > summary p,
    div[data-testid="stExpander"] details > summary svg {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    div[data-testid="stExpander"] details > div {
        background: rgba(255, 255, 255, 0.94);
        padding: 0.35rem 0.4rem 0.4rem 0.4rem;
    }

    div[data-baseweb="textarea"] > div,
    div[data-testid="stTextArea"] textarea {
        background: #ffffff !important;
        color: #102028 !important;
    }

    div[data-testid="stTextArea"] textarea::placeholder {
        color: rgba(16, 32, 40, 0.58) !important;
    }

    .table-shell {
        background: #ffffff;
        border: 1px solid rgba(16, 32, 40, 0.12);
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 8px 22px rgba(16, 32, 40, 0.06);
        margin: 0.35rem 0 0.75rem 0;
    }

    .table-shell table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.94rem;
        table-layout: fixed;
    }

    .table-shell thead th {
        background: #697d8a;
        color: #ffffff;
        font-weight: 400;
        text-align: left;
        padding: 0.52rem 0.8rem;
        border-bottom: 1px solid rgba(16, 32, 40, 0.10);
    }

    .table-shell tbody td {
        color: #163243;
        padding: 0.48rem 0.8rem;
        border-bottom: 1px solid rgba(16, 32, 40, 0.08);
        vertical-align: top;
        line-height: 1.25;
        word-break: break-word;
        white-space: normal;
    }

    .table-shell tbody tr:nth-child(odd) {
        background: #f7f8fa;
    }

    .table-shell tbody tr:nth-child(even) {
        background: #eceff3;
    }

    [data-testid="stSidebar"] .table-shell table {
        font-size: 0.88rem;
    }

    .soft-note {
        background: #e6e8ec;
        color: #163243;
        border-radius: 12px;
        padding: 0.8rem 0.95rem;
        border: 1px solid rgba(16, 32, 40, 0.10);
        margin: 0.35rem 0 0.75rem 0;
    }

    .artifact-note {
        background: rgba(255, 255, 255, 0.92);
        color: #163243;
        border: 1px solid rgba(51, 91, 132, 0.16);
        border-left: 6px solid #335b84;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        margin: 0.3rem 0 0.9rem 0;
        box-shadow: 0 8px 22px rgba(16, 32, 40, 0.06);
    }

    .artifact-note strong {
        display: block;
        font-size: 0.98rem;
        margin-bottom: 0.18rem;
    }

    .artifact-note code {
        background: rgba(51, 91, 132, 0.08);
        color: #163243;
        padding: 0.18rem 0.4rem;
        border-radius: 8px;
        font-size: 0.84rem;
        word-break: break-all;
    }

    .plain-alert {
        background: transparent;
        color: #8b1e1e;
        padding: 0.1rem 0;
        margin: 0.2rem 0 0.6rem 0;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def find_artifact(filename: str) -> Path | None:
    candidate = ARTIFACT_DIR / filename
    if candidate.exists():
        return candidate
    return None


def list_available_pipeline_files() -> list[str]:
    available = []
    for filename in PIPELINE_FILE_ORDER:
        if find_artifact(filename):
            available.append(filename)
    return available


def format_model_name(filename: str) -> str:
    if filename == "best_model_pipeline.pkl":
        return "Best Model"
    return (
        filename.replace("_pipeline.pkl", "")
        .replace("_", " ")
        .title()
        .replace("Svm", "SVM")
    )


@st.cache_resource
def load_pipeline_model(model_file: str):
    model_path = find_artifact(model_file)
    if model_path is None:
        raise FileNotFoundError(f"Could not find model artifact: {model_file}")
    return joblib.load(model_path), str(model_path)


@st.cache_data
def load_best_model_metadata() -> dict:
    metadata_path = find_artifact("best_model_metadata.pkl")
    if metadata_path is None:
        return {}
    with open(metadata_path, "rb") as handle:
        return pickle.load(handle)


@st.cache_data
def load_benchmark_results():
    for filename in RESULT_FILE_ORDER:
        result_path = find_artifact(filename)
        if result_path:
            with open(result_path, "rb") as handle:
                return pickle.load(handle), str(result_path)
    return {}, ""


def select_default_model_file() -> str | None:
    if find_artifact("best_model_pipeline.pkl"):
        return "best_model_pipeline.pkl"

    metadata = load_best_model_metadata()
    best_slug = metadata.get("best_model_slug")
    if best_slug:
        best_file = f"{best_slug}_pipeline.pkl"
        if find_artifact(best_file):
            return best_file

    results, _ = load_benchmark_results()
    model_name_to_file = {
        "Linear SVM": "linear_svm_pipeline.pkl",
        "Logistic Regression": "logistic_regression_pipeline.pkl",
        "Naive Bayes": "naive_bayes_pipeline.pkl",
        "Random Forest": "random_forest_pipeline.pkl",
    }
    best_name = None
    best_cv = -1.0
    for model_name, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        cv_score = float(metrics.get("cv_mean", -1))
        if model_name in model_name_to_file and cv_score > best_cv:
            best_cv = cv_score
            best_name = model_name
    if best_name:
        return model_name_to_file[best_name]

    available = list_available_pipeline_files()
    return available[0] if available else None


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

CONTRACTION_MAP = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
}

ABBREVIATION_MAP = {
    "u.s.": "us",
    "u.k.": "uk",
    "govt": "government",
    "w/": "with",
    "w/o": "without",
    "dept": "department",
    "info": "information",
}


def replace_dictionary_terms(text: str, mapping: dict[str, str]) -> str:
    for source, target in mapping.items():
        text = re.sub(rf"\b{re.escape(source)}\b", target, text)
    return text


def normalize_text(text: str) -> str:
    text = str(text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = replace_dictionary_terms(text, CONTRACTION_MAP)
    text = replace_dictionary_terms(text, ABBREVIATION_MAP)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_content(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text: str) -> list[str]:
    return [token for token in word_tokenize(text) if token.isalpha()]


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in stop_words and len(token) > 2]


def get_wordnet_pos(tag: str):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize_with_pos(tokens: list[str]) -> list[str]:
    tagged_tokens = pos_tag(tokens)
    return [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in tagged_tokens]


def preprocess_text_for_modeling(text: str) -> dict:
    normalized_text = normalize_text(text)
    clean_text = clean_text_content(normalized_text)
    tokens = tokenize_text(clean_text)
    filtered_tokens = remove_stopwords(tokens)
    lemmatized_tokens = lemmatize_with_pos(filtered_tokens)
    return {
        "raw_text": text,
        "normalized_text": normalized_text,
        "clean_text": clean_text,
        "tokens": tokens,
        "tokens_no_stopwords": filtered_tokens,
        "lemmatized_tokens": lemmatized_tokens,
        "cleaned_text": " ".join(lemmatized_tokens),
    }


def preprocess_with_trace(text: str) -> dict:
    return preprocess_text_for_modeling(text)


def validate_text_input(
    text: str,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> tuple[bool, str]:
    stripped = text.strip()
    if not stripped:
        return False, "Please enter article text before running prediction."
    if len(stripped) < min_chars:
        return False, f"Article text is too short. Minimum length is {min_chars} characters."
    if len(stripped) > max_chars:
        return False, f"Article text is too long. Maximum length is {max_chars} characters."
    return True, ""


def to_probability_vector(model, cleaned_text: str):
    if hasattr(model, "predict_proba"):
        return model.predict_proba([cleaned_text])[0]
    if hasattr(model, "decision_function"):
        decision_score = float(model.decision_function([cleaned_text])[0])
        real_probability = 1 / (1 + np.exp(-decision_score))
        return np.array([1 - real_probability, real_probability])
    prediction = int(model.predict([cleaned_text])[0])
    return np.array([1.0, 0.0]) if prediction == 0 else np.array([0.0, 1.0])


def predict_news(text: str, threshold: float, model_file: str):
    trace = preprocess_with_trace(text)
    model, model_path = load_pipeline_model(model_file)
    probability = to_probability_vector(model, trace["cleaned_text"])

    real_probability = float(probability[1])
    prediction = 1 if real_probability >= threshold else 0
    result_label = LABEL_MAP[prediction]
    confidence = real_probability if prediction == 1 else 1 - real_probability

    return {
        "label": result_label,
        "confidence": confidence,
        "probabilities": probability,
        "trace": trace,
        "model_file": model_file,
        "model_path": model_path,
    }


def extract_text_from_url(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        )
    }

    if trafilatura is not None:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_links=False,
                include_tables=False,
                include_images=False,
                favor_precision=True,
            )
            if extracted and extracted.strip():
                return extracted.strip()

    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    article_paragraphs = []
    for node in soup.find_all("article"):
        article_paragraphs.extend(p.get_text(" ", strip=True) for p in node.find_all("p"))

    main_paragraphs = []
    for node in soup.find_all("main"):
        main_paragraphs.extend(p.get_text(" ", strip=True) for p in node.find_all("p"))

    paragraphs = article_paragraphs or main_paragraphs
    if not paragraphs:
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    return "\n".join(p for p in paragraphs if p).strip()


def strip_html(value: str) -> str:
    if not value:
        return ""
    text = BeautifulSoup(value, "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def get_feed_node_text(item, *tag_names: str) -> str:
    for tag_name in tag_names:
        node = item.find(tag_name)
        if node:
            return node.get_text(" ", strip=True)
    return ""


@st.cache_data(show_spinner=False, ttl=3600)
def load_fake_dataset() -> pd.DataFrame:
    dataset_path = BASE_DIR / "Dataset" / "fake.csv"
    fake_df = pd.read_csv(dataset_path)
    fake_df["title"] = fake_df["title"].fillna("")
    fake_df["text"] = fake_df["text"].fillna("")
    fake_df["combined_text"] = (fake_df["title"] + ". " + fake_df["text"]).str.strip()
    fake_df["search_text"] = fake_df["combined_text"].str.lower()
    return fake_df


@st.cache_data(show_spinner=False, ttl=3600)
def load_true_dataset() -> pd.DataFrame:
    dataset_path = BASE_DIR / "Dataset" / "true.csv"
    true_df = pd.read_csv(dataset_path)
    true_df["title"] = true_df["title"].fillna("")
    true_df["text"] = true_df["text"].fillna("")
    true_df["combined_text"] = (true_df["title"] + ". " + true_df["text"]).str.strip()
    true_df["search_text"] = true_df["combined_text"].str.lower()
    return true_df


def build_dataset_samples(
    dataframe: pd.DataFrame,
    source_key: str,
    limit: int,
    title_fallback: str,
) -> list[dict]:
    keywords = NEWS_SOURCE_CATALOG[source_key]["fake_keywords"]
    keyword_pattern = "|".join(re.escape(keyword.lower()) for keyword in keywords)

    if keyword_pattern:
        themed_matches = dataframe[dataframe["search_text"].str.contains(keyword_pattern, regex=True, na=False)]
    else:
        themed_matches = dataframe.iloc[0:0]

    themed_count = min(limit, len(themed_matches))
    selected_rows = themed_matches.sample(n=themed_count, random_state=42) if themed_count else dataframe.iloc[0:0]

    if themed_count < limit:
        remaining = limit - themed_count
        remainder_pool = dataframe.drop(selected_rows.index, errors="ignore")
        if not remainder_pool.empty:
            fallback_rows = remainder_pool.sample(n=min(remaining, len(remainder_pool)), random_state=42)
            selected_rows = pd.concat([selected_rows, fallback_rows], ignore_index=False)

    samples = []
    for _, row in selected_rows.head(limit).iterrows():
        samples.append(
            {
                "title": row["title"] or title_fallback,
                "text": row["combined_text"],
                "url": "",
                "published": str(row.get("date", "")).strip(),
            }
        )
    return samples


def predict_sample_label(text: str, model_file: str) -> str | None:
    try:
        prediction = predict_news(text, DEFAULT_THRESHOLD, model_file)
        return prediction["label"]
    except Exception:
        return None


def sample_matches_expected_label(
    sample: dict,
    expected_label: str,
    model_file: str,
    min_chars: int,
    max_chars: int,
) -> bool:
    is_valid, _ = validate_text_input(sample["text"], min_chars, max_chars)
    if not is_valid:
        return False
    predicted_label = predict_sample_label(sample["text"], model_file)
    return predicted_label == expected_label


def filter_samples_by_expected_label(
    candidates: list[dict],
    expected_label: str,
    model_file: str,
    min_chars: int,
    max_chars: int,
    limit: int,
) -> list[dict]:
    matched_samples = []
    seen_signatures = set()

    for sample in candidates:
        signature = (sample.get("title", ""), sample.get("text", "")[:180])
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        if sample_matches_expected_label(sample, expected_label, model_file, min_chars, max_chars):
            matched_samples.append(sample)
        if len(matched_samples) >= limit:
            break

    return matched_samples


def generate_backup_samples(source_key: str, sample_type: str, count: int) -> list[dict]:
    source_label = NEWS_SOURCE_CATALOG[source_key]["label"]
    keywords = NEWS_SOURCE_CATALOG[source_key]["fake_keywords"]
    topic = keywords[0].title() if keywords else source_label
    samples = []

    for index in range(count):
        if sample_type == "real":
            title = f"{source_label} verified report {index + 1}"
            text = (
                f"{source_label} coverage reports verified developments related to {topic}. "
                f"Officials, public statements, and named institutions are cited in this article. "
                f"The report describes confirmed events, policy responses, and contextual details for readers."
            )
        else:
            title = f"{source_label} fake alert sample {index + 1}"
            text = (
                f"Breaking rumors claim secret actors manipulated {topic} through hidden operations with no verified evidence. "
                f"Anonymous insiders, dramatic language, and unsupported accusations are used to push the story. "
                f"The article exaggerates events and presents conspiracy-style claims without trustworthy sources."
            )

        samples.append(
            {
                "title": title,
                "text": text,
                "url": "",
                "published": "Generated fallback",
            }
        )
    return samples


def build_fake_samples(
    source_key: str,
    model_file: str,
    min_chars: int,
    max_chars: int,
    limit: int = FAKE_PREDICT_SAMPLE_LIMIT,
) -> list[dict]:
    fake_candidates = build_dataset_samples(
        dataframe=load_fake_dataset(),
        source_key=source_key,
        limit=max(limit * 60, 300),
        title_fallback="Untitled fake sample",
    )
    valid_fake_samples = filter_samples_by_expected_label(
        candidates=fake_candidates,
        expected_label="Fake News",
        model_file=model_file,
        min_chars=min_chars,
        max_chars=max_chars,
        limit=limit,
    )

    if len(valid_fake_samples) < limit:
        generated_fake_samples = generate_backup_samples(source_key, "fake", max((limit - len(valid_fake_samples)) * 12, 24))
        validated_generated_fake_samples = filter_samples_by_expected_label(
            candidates=generated_fake_samples,
            expected_label="Fake News",
            model_file=model_file,
            min_chars=min_chars,
            max_chars=max_chars,
            limit=limit - len(valid_fake_samples),
        )
        valid_fake_samples.extend(validated_generated_fake_samples)

    return valid_fake_samples[:limit]


def build_real_fallback_samples(
    source_key: str,
    model_file: str,
    min_chars: int,
    max_chars: int,
    limit: int,
) -> list[dict]:
    real_candidates = build_dataset_samples(
        dataframe=load_true_dataset(),
        source_key=source_key,
        limit=max(limit * 60, 300),
        title_fallback="Untitled real sample",
    )
    valid_real_samples = filter_samples_by_expected_label(
        candidates=real_candidates,
        expected_label="Real News",
        model_file=model_file,
        min_chars=min_chars,
        max_chars=max_chars,
        limit=limit,
    )

    if len(valid_real_samples) < limit:
        generated_real_samples = generate_backup_samples(source_key, "real", max((limit - len(valid_real_samples)) * 12, 24))
        validated_generated_real_samples = filter_samples_by_expected_label(
            candidates=generated_real_samples,
            expected_label="Real News",
            model_file=model_file,
            min_chars=min_chars,
            max_chars=max_chars,
            limit=limit - len(valid_real_samples),
        )
        valid_real_samples.extend(validated_generated_real_samples)

    return valid_real_samples[:limit]


def parse_rss_entries(feed_url: str) -> list[dict]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        )
    }
    response = requests.get(feed_url, headers=headers, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "xml")
    items = soup.find_all("item") or soup.find_all("entry")

    entries = []
    for item in items:
        link_node = item.find("link")
        link = ""
        if link_node:
            link = (link_node.get("href") or link_node.get_text(" ", strip=True) or "").strip()

        title = strip_html(get_feed_node_text(item, "title"))
        summary = strip_html(get_feed_node_text(item, "description", "summary", "content"))
        published = strip_html(get_feed_node_text(item, "pubDate", "published", "updated"))
        if title and (link or summary):
            entries.append(
                {
                    "title": title,
                    "summary": summary,
                    "url": link,
                    "published": published,
                }
            )
    return entries


def scrape_homepage_entries(homepage_url: str, max_candidates: int = 20) -> list[dict]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        )
    }
    response = requests.get(homepage_url, headers=headers, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    homepage_host = urlparse(homepage_url).netloc.replace("www.", "")
    entries = []
    seen_urls = set()

    for link in soup.find_all("a", href=True):
        href = (link.get("href") or "").strip()
        title = strip_html(link.get_text(" ", strip=True))
        if not href or not title or len(title) < 25:
            continue

        absolute_url = urljoin(homepage_url, href)
        parsed_url = urlparse(absolute_url)
        article_host = parsed_url.netloc.replace("www.", "")
        path = parsed_url.path.lower()

        if parsed_url.scheme not in {"http", "https"}:
            continue
        if homepage_host and article_host and homepage_host not in article_host and article_host not in homepage_host:
            continue
        if absolute_url in seen_urls or path in {"", "/"}:
            continue
        if any(token in path for token in ["/tag/", "/topic/", "/author/", "/video/", "/gallery/", "/live/"]):
            continue

        seen_urls.add(absolute_url)
        entries.append(
            {
                "title": title,
                "summary": "",
                "url": absolute_url,
                "published": "",
            }
        )
        if len(entries) >= max_candidates:
            break

    return entries


def entry_matches_source(entry: dict, source_key: str) -> bool:
    keywords = NEWS_SOURCE_CATALOG[source_key]["fake_keywords"]
    haystack = " ".join(
        [
            entry.get("title", ""),
            entry.get("summary", ""),
            entry.get("url", ""),
        ]
    ).lower()
    return any(keyword.lower() in haystack for keyword in keywords)


def build_real_sample_text(entry: dict, require_article_fetch: bool) -> str:
    fallback_text = f"{entry['title']}. {entry['summary']}".strip()
    if require_article_fetch and entry["url"]:
        try:
            extracted_text = extract_text_from_url(entry["url"])
            is_valid, _ = validate_text_input(extracted_text)
            if is_valid:
                return extracted_text
        except Exception:
            pass
    return fallback_text


def fetch_source_bundle(source_key: str, model_file: str, min_chars: int, max_chars: int) -> dict:
    source_config = NEWS_SOURCE_CATALOG[source_key]
    feed_entries = []
    load_notes = []

    for feed_url in source_config["feeds"]:
        try:
            feed_entries.extend(parse_rss_entries(feed_url))
        except Exception as exc:
            load_notes.append(f"Feed fallback used for {feed_url}: {exc}")

    if len(feed_entries) < REAL_SOURCE_SAMPLE_LIMIT:
        try:
            homepage_entries = scrape_homepage_entries(source_config["homepage"], max_candidates=25)
            feed_entries.extend(homepage_entries)
        except Exception as exc:
            load_notes.append(f"Homepage fallback failed for {source_config['homepage']}: {exc}")

    relevant_entries = [entry for entry in feed_entries if entry_matches_source(entry, source_key)]
    fallback_entries = [entry for entry in feed_entries if not entry_matches_source(entry, source_key)]
    ordered_entries = relevant_entries + fallback_entries

    real_samples = []
    seen_keys = set()
    for entry in ordered_entries:
        unique_key = entry["url"] or entry["title"]
        if not unique_key or unique_key in seen_keys:
            continue
        seen_keys.add(unique_key)

        sample_text = build_real_sample_text(
            entry,
            require_article_fetch=True,
        )
        real_candidate = {
            "title": entry["title"],
            "text": sample_text,
            "url": entry["url"],
            "published": entry["published"],
        }

        if not sample_matches_expected_label(real_candidate, "Real News", model_file, min_chars, max_chars):
            continue

        real_samples.append(real_candidate)
        if len(real_samples) >= REAL_SOURCE_SAMPLE_LIMIT:
            break

    if len(real_samples) < REAL_SOURCE_SAMPLE_LIMIT:
        existing_titles = {sample["title"] for sample in real_samples}
        fallback_real_samples = build_real_fallback_samples(
            source_key=source_key,
            model_file=model_file,
            min_chars=min_chars,
            max_chars=max_chars,
            limit=REAL_SOURCE_SAMPLE_LIMIT - len(real_samples),
        )
        for sample in fallback_real_samples:
            if sample["title"] in existing_titles:
                continue
            real_samples.append(sample)
            existing_titles.add(sample["title"])
            if len(real_samples) >= REAL_SOURCE_SAMPLE_LIMIT:
                break

    if len(real_samples) < REAL_SOURCE_SAMPLE_LIMIT:
        real_samples.extend(
            filter_samples_by_expected_label(
                candidates=generate_backup_samples(
                    source_key=source_key,
                    sample_type="real",
                    count=max((REAL_SOURCE_SAMPLE_LIMIT - len(real_samples)) * 6, 12),
                ),
                expected_label="Real News",
                model_file=model_file,
                min_chars=min_chars,
                max_chars=max_chars,
                limit=REAL_SOURCE_SAMPLE_LIMIT - len(real_samples),
            )
        )

    fake_samples = build_fake_samples(source_key, model_file=model_file, min_chars=min_chars, max_chars=max_chars)
    return {
        "source_key": source_key,
        "model_file": model_file,
        "source_label": source_config["label"],
        "homepage": source_config["homepage"],
        "real_samples": real_samples[:REAL_SOURCE_SAMPLE_LIMIT],
        "fake_samples": fake_samples,
        "load_notes": load_notes[:3],
    }


def render_result_card(prediction: dict) -> None:
    badge_class = "real" if prediction["label"] == "Real News" else "fake"
    fill_color = REAL_COLOR if prediction["label"] == "Real News" else FAKE_COLOR
    confidence_pct = prediction["confidence"] * 100
    st.markdown(
        f"""
        <div class="result-card {badge_class}">
            <div class="result-label">{prediction['label']}</div>
            <div class="result-meta">Confidence: {confidence_pct:.2f}%</div>
            <div class="result-meta">Model: {format_model_name(prediction['model_file'])}</div>
            <div class="confidence-track">
                <div class="confidence-fill" style="width:{confidence_pct:.1f}%; background:{fill_color};">
                    <span class="confidence-value" style="border:1px solid {fill_color};">{confidence_pct:.2f}%</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    probability_df = pd.DataFrame(
        {
            "Class": ["Fake News", "Real News"],
            "Probability": prediction["probabilities"],
        }
    )
    render_styled_table(probability_df, formatters={"Probability": "{:.4f}"})


def render_styled_table(dataframe: pd.DataFrame, formatters: dict | None = None) -> None:
    display_df = dataframe.copy()
    if formatters:
        for column, formatter in formatters.items():
            if column not in display_df.columns:
                continue
            if callable(formatter):
                display_df[column] = display_df[column].map(
                    lambda value: "" if pd.isna(value) else formatter(value)
                )
            else:
                display_df[column] = display_df[column].map(
                    lambda value: "" if pd.isna(value) else formatter.format(value)
                )

    display_df = display_df.fillna("")
    table_html = display_df.to_html(index=False, escape=True, border=0)
    st.markdown(f'<div class="table-shell">{table_html}</div>', unsafe_allow_html=True)


def render_preprocessing_trace(trace: dict) -> None:
    with st.expander("Show Professor-Aligned Preprocessing Trace", expanded=True):
        stage_rows = [
            ("Text Normalization", trace["normalized_text"][:500]),
            ("Text Cleaning", trace["clean_text"][:500]),
            ("Tokenization", ", ".join(trace["tokens"][:40])),
            ("Remove Stopwords", ", ".join(trace["tokens_no_stopwords"][:40])),
            ("POS-Aware Lemmatization", ", ".join(trace["lemmatized_tokens"][:40])),
            ("Final Modeling Text", trace["cleaned_text"][:500]),
        ]
        stage_df = pd.DataFrame(stage_rows, columns=["Pipeline Step", "Preview"])
        render_styled_table(stage_df)
        st.caption(
            f"Token counts: raw tokens={len(trace['tokens'])}, after stopword removal={len(trace['tokens_no_stopwords'])}, lemmas={len(trace['lemmatized_tokens'])}"
        )


def render_pipeline_tab() -> None:
    selected_image_path = BASE_DIR / "assets" / "PipeLine.png"
    if not selected_image_path.exists():
        st.warning(f"Pipeline image not found: `{selected_image_path.name}`")
        return

    left_spacer, image_col, right_spacer = st.columns([2, 6, 2])
    with image_col:
        st.image(str(selected_image_path), use_column_width=True)


def parse_notebook_table_output(cell: dict, table_index: int = 0) -> pd.DataFrame:
    html_outputs = []
    for output in cell.get("outputs", []):
        data = output.get("data", {})
        if "text/html" in data:
            html_outputs.append("".join(data["text/html"]))

    if table_index >= len(html_outputs):
        return pd.DataFrame()

    table_df = pd.read_html(StringIO(html_outputs[table_index]))[0]
    return table_df.loc[:, ~table_df.columns.astype(str).str.startswith("Unnamed")].copy()


def parse_notebook_stream_text(cell: dict, stream_index: int = 0) -> str:
    streams = [
        "".join(output.get("text", []))
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    ]
    if stream_index >= len(streams):
        return ""
    return streams[stream_index].strip()


def parse_notebook_png_output(cell: dict, image_index: int = 0) -> bytes:
    images = []
    for output in cell.get("outputs", []):
        data = output.get("data", {})
        if "image/png" in data:
            images.append(base64.b64decode("".join(data["image/png"])))
    if image_index >= len(images):
        return b""
    return images[image_index]


@st.cache_data(show_spinner=False)
def load_notebook_benchmark_context() -> dict:
    notebook_path = BASE_DIR / "News_Classification.ipynb"
    if not notebook_path.exists():
        return {}

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])

    try:
        tuned_cell = cells[NOTEBOOK_BENCHMARK_CELLS["tuned_comparison"]]
        evaluation_cell = cells[NOTEBOOK_BENCHMARK_CELLS["evaluation"]]
        return {
            "baseline_comparison": parse_notebook_table_output(cells[NOTEBOOK_BENCHMARK_CELLS["baseline_comparison"]]),
            "robust_cv": parse_notebook_table_output(cells[NOTEBOOK_BENCHMARK_CELLS["robust_cv"]]),
            "tuned_comparison": parse_notebook_table_output(tuned_cell, table_index=0),
            "best_model_metadata_table": parse_notebook_table_output(tuned_cell, table_index=1),
            "stage_comparison": parse_notebook_table_output(cells[NOTEBOOK_BENCHMARK_CELLS["stage_comparison"]]),
            "selection_summary": parse_notebook_stream_text(tuned_cell),
            "classification_report_text": parse_notebook_stream_text(evaluation_cell),
            "evaluation_figure_png": parse_notebook_png_output(evaluation_cell),
            "stage_comparison_figure_png": parse_notebook_png_output(cells[NOTEBOOK_BENCHMARK_CELLS["stage_comparison"]]),
        }
    except (IndexError, ValueError):
        return {}


@st.cache_data(show_spinner=False)
def build_benchmark_modeling_frame() -> pd.DataFrame:
    fake_df = pd.read_csv(BASE_DIR / "Dataset" / "fake.csv")
    fake_df["label"] = 0
    true_df = pd.read_csv(BASE_DIR / "Dataset" / "true.csv")
    true_df["label"] = 1

    model_df = pd.concat([fake_df, true_df], ignore_index=True)
    model_df["title"] = model_df["title"].fillna("")
    model_df["text"] = model_df["text"].fillna("")
    model_df["raw_text"] = (model_df["title"] + ". " + model_df["text"]).str.strip()
    model_df = model_df[model_df["raw_text"].str.strip().ne("")].reset_index(drop=True)
    model_df["cleaned_text"] = model_df["raw_text"].map(
        lambda value: preprocess_text_for_modeling(value)["cleaned_text"]
    )
    model_df = model_df[model_df["cleaned_text"].str.strip().ne("")].reset_index(drop=True)
    return model_df[["cleaned_text", "label"]]


@st.cache_data(show_spinner=False)
def recreate_notebook_test_split() -> tuple[pd.Series, pd.Series]:
    model_df = build_benchmark_modeling_frame()
    _, X_test, _, y_test = train_test_split(
        model_df["cleaned_text"],
        model_df["label"],
        test_size=0.20,
        random_state=42,
        stratify=model_df["label"],
    )
    return X_test.reset_index(drop=True), y_test.reset_index(drop=True)


def resolve_best_benchmark_model_file() -> str | None:
    metadata = load_best_model_metadata()
    best_slug = metadata.get("best_model_slug")
    if best_slug:
        return MODEL_SLUG_TO_PIPELINE_FILE.get(best_slug)
    return select_default_model_file()


@st.cache_data(show_spinner=False)
def evaluate_best_benchmark_model(model_file: str) -> dict:
    model_path = find_artifact(model_file)
    if model_path is None:
        return {}

    pipeline = joblib.load(model_path)
    X_test, y_test = recreate_notebook_test_split()
    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline, "predict_proba"):
        y_score = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        decision_score = pipeline.decision_function(X_test)
        y_score = 1 / (1 + np.exp(-decision_score))
    else:
        y_score = y_pred.astype(float)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_score)
    return {
        "model_file": model_file,
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_score": y_score.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision_curve": precision_vals.tolist(),
        "recall_curve": recall_vals.tolist(),
        "roc_auc": float(auc(fpr, tpr)),
        "pr_auc": float(auc(recall_vals, precision_vals)),
        "classification_report_text": classification_report(y_test, y_pred, digits=4),
    }


def build_legacy_summary_frame(results: dict) -> pd.DataFrame:
    rows = []
    for model_name, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "Model": model_name,
                "Stage": str(metrics.get("stage", "")).title(),
                "Search Method": metrics.get("search_method", ""),
                "Accuracy": float(metrics.get("accuracy", 0)),
                "Precision": float(metrics.get("precision", 0)),
                "Recall": float(metrics.get("recall", 0)),
                "F1 Score": float(metrics.get("f1", 0)),
                "CV Mean": float(metrics.get("cv_mean", 0)),
                "CV Std": float(metrics.get("cv_std", 0) or 0),
                "Best Parameters": str(metrics.get("best_params", "")),
            }
        )
    return pd.DataFrame(rows).sort_values(["CV Mean", "F1 Score"], ascending=False)


def extract_metric_results(raw_results: dict) -> dict:
    if isinstance(raw_results, dict) and "legacy_results" in raw_results:
        legacy_results = raw_results.get("legacy_results", {})
        if isinstance(legacy_results, dict):
            return legacy_results
    return raw_results if isinstance(raw_results, dict) else {}


def benchmark_payload_context(raw_results: dict) -> dict:
    if not (isinstance(raw_results, dict) and raw_results.get("artifact_version")):
        return {}

    context = {}
    for key in ["baseline_comparison", "robust_cv", "tuned_comparison", "best_model_metadata_table", "stage_comparison"]:
        records = raw_results.get(key, [])
        if isinstance(records, list):
            context[key] = pd.DataFrame(records)
    for key in ["selection_summary", "classification_report_text"]:
        if raw_results.get(key):
            context[key] = raw_results.get(key, "")
    for key in ["evaluation_figure_png", "stage_comparison_figure_png"]:
        if raw_results.get(key):
            context[key] = raw_results[key]
    if isinstance(raw_results.get("best_model_evaluation"), dict):
        context["best_model_evaluation"] = raw_results["best_model_evaluation"]
    return context


def parse_best_model_name(selection_summary: str, metadata_table: pd.DataFrame) -> str:
    for line in selection_summary.splitlines():
        if line.startswith("Selected final tuned model:"):
            return line.split(":", 1)[1].strip()
    if not metadata_table.empty and {"Item", "Value"}.issubset(metadata_table.columns):
        matched = metadata_table.loc[metadata_table["Item"] == "best_model_name", "Value"]
        if not matched.empty:
            return str(matched.iloc[0])
    metadata = load_best_model_metadata()
    return str(metadata.get("best_model_name", "Best Model"))


def lookup_model_metrics(dataframe: pd.DataFrame, model_name: str) -> dict:
    if dataframe.empty or "Model" not in dataframe.columns:
        return {}
    matched = dataframe[dataframe["Model"] == model_name]
    if matched.empty:
        return {}
    return matched.iloc[0].to_dict()


def render_notebook_figure(image_bytes: bytes, caption: str | None = None) -> None:
    if not image_bytes:
        return
    st.image(image_bytes, use_column_width=True)
    if caption:
        st.caption(caption)


def parse_classification_report_tables(report_text: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_rows = []
    summary_rows = []
    summary_labels = {"accuracy", "macro avg", "weighted avg", "micro avg", "samples avg"}
    label_map = {"0": "Fake News", "1": "Real News", "fake": "Fake News", "real": "Real News"}

    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("precision"):
            continue

        parts = re.split(r"\s{2,}", line)
        if not parts:
            continue

        label = parts[0].strip()
        normalized_label = label.lower()

        if normalized_label == "accuracy" and len(parts) >= 3:
            summary_rows.append(
                {
                    "Summary": "Accuracy",
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "F1 Score": float(parts[1]),
                    "Support": int(float(parts[2])),
                }
            )
            continue

        if len(parts) < 5:
            continue

        row = {
            "Label": label_map.get(normalized_label, label.title()),
            "Precision": float(parts[1]),
            "Recall": float(parts[2]),
            "F1 Score": float(parts[3]),
            "Support": int(float(parts[4])),
        }

        if normalized_label in summary_labels:
            summary_rows.append({"Summary": row["Label"], **{key: row[key] for key in ["Precision", "Recall", "F1 Score", "Support"]}})
        else:
            class_rows.append(row)

    return pd.DataFrame(class_rows), pd.DataFrame(summary_rows)


def render_classification_report_block(report_text: str) -> None:
    st.markdown("#### Classification Report")
    class_df, summary_df = parse_classification_report_tables(report_text)
    if class_df.empty and summary_df.empty:
        st.code(report_text.strip(), language="text")
        return

    formatter = {
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1 Score": "{:.4f}",
    }
    if not summary_df.empty and "Summary" in summary_df.columns:
        summary_lookup = {
            str(row["Summary"]).strip().lower(): row
            for _, row in summary_df.iterrows()
        }
        metric_cols = st.columns(4)
        accuracy_row = summary_lookup.get("accuracy")
        macro_row = summary_lookup.get("Macro Avg".lower())
        weighted_row = summary_lookup.get("Weighted Avg".lower())
        total_support = ""
        if accuracy_row is not None:
            metric_cols[0].metric("Accuracy", f"{float(accuracy_row['F1 Score']):.4f}")
            total_support = f"{int(accuracy_row['Support']):,}"
        if macro_row is not None:
            metric_cols[1].metric("Macro F1", f"{float(macro_row['F1 Score']):.4f}")
        if weighted_row is not None:
            metric_cols[2].metric("Weighted F1", f"{float(weighted_row['F1 Score']):.4f}")
        if total_support:
            metric_cols[3].metric("Support", total_support)

    if not class_df.empty:
        st.markdown("##### Per-Class Metrics")
        render_styled_table(class_df, formatters=formatter)
    if not summary_df.empty:
        st.markdown("##### Summary Metrics")
        render_styled_table(summary_df, formatters=formatter)


def build_dashboard_winners_table(
    stage_comparison_df: pd.DataFrame,
    baseline_comparison_df: pd.DataFrame,
    tuned_comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    comparison_df = stage_comparison_df.copy()
    if comparison_df.empty:
        frames = []
        if not baseline_comparison_df.empty:
            baseline_df = baseline_comparison_df.copy()
            baseline_df["Stage"] = "Baseline"
            frames.append(baseline_df)
        if not tuned_comparison_df.empty:
            tuned_df = tuned_comparison_df.copy()
            tuned_df["Stage"] = "Tuned"
            frames.append(tuned_df)
        if frames:
            comparison_df = pd.concat(frames, ignore_index=True)

    required_columns = ["Model", "Stage", "Accuracy", "Precision", "Recall", "F1 Score"]
    if comparison_df.empty or not set(required_columns).issubset(comparison_df.columns):
        return pd.DataFrame()

    ranking_df = comparison_df[required_columns].copy()
    for metric in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        ranking_df[metric] = pd.to_numeric(ranking_df[metric], errors="coerce")
    ranking_df["Stage"] = ranking_df["Stage"].astype(str).str.title()
    ranking_df = ranking_df.sort_values(
        by=["Model", "F1 Score", "Accuracy", "Precision", "Recall"],
        ascending=[True, False, False, False, False],
    )
    winners_df = ranking_df.groupby("Model", as_index=False).first()
    winners_df.insert(1, "Best Version", winners_df["Model"] + " " + winners_df["Stage"])
    winners_df = winners_df.rename(columns={"Model": "Model Family", "Stage": "Selected Stage"})
    winners_df = winners_df.sort_values(by=["F1 Score", "Accuracy"], ascending=[False, False]).reset_index(drop=True)
    return winners_df[
        ["Model Family", "Best Version", "Selected Stage", "Accuracy", "Precision", "Recall", "F1 Score"]
    ]


def render_best_model_evaluation(evaluation: dict, best_model_name: str) -> None:
    if not evaluation:
        st.warning("Best-model evaluation could not be reconstructed from the saved artifacts.")
        return

    y_test = np.array(evaluation["y_test"])
    y_pred = np.array(evaluation["y_pred"])
    fpr = np.array(evaluation["fpr"])
    tpr = np.array(evaluation["tpr"])
    recall_vals = np.array(evaluation["recall_curve"])
    precision_vals = np.array(evaluation["precision_curve"])

    metric_cols = st.columns(4)
    metric_cols[0].metric("Final Tuned Model", best_model_name)
    metric_cols[1].metric("ROC AUC", f"{evaluation['roc_auc']:.4f}")
    metric_cols[2].metric("PR AUC", f"{evaluation['pr_auc']:.4f}")
    metric_cols[3].metric("Test Samples", f"{len(y_test):,}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Fake", "Real"],
        cmap="Blues",
        colorbar=False,
        ax=axes[0],
    )
    axes[0].set_title(f"{best_model_name} Confusion Matrix", fontsize=14, fontweight="bold")

    axes[1].plot(fpr, tpr, color=BENCHMARK_REAL_COLOR, linewidth=2.5, label=f"AUC = {evaluation['roc_auc']:.4f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set_title("ROC Curve", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25, linestyle="--")

    axes[2].plot(
        recall_vals,
        precision_vals,
        color=BENCHMARK_FAKE_COLOR,
        linewidth=2.5,
        label=f"AUC = {evaluation['pr_auc']:.4f}",
    )
    axes[2].set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].legend(frameon=False)
    axes[2].grid(alpha=0.25, linestyle="--")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    render_classification_report_block(evaluation["classification_report_text"])


def render_metric_heatmap(dataframe: pd.DataFrame, title: str) -> None:
    if dataframe.empty:
        return
    metric_columns = [column for column in ["Accuracy", "Precision", "Recall", "F1 Score", "CV Mean"] if column in dataframe.columns]
    if not metric_columns:
        return

    heatmap_df = dataframe.set_index("Model")[metric_columns]
    dashboard_cmap = LinearSegmentedColormap.from_list(
        "dashboard_metrics",
        ["#eef6f4", "#9fd4bf", "#3CB371", "#0f6096"],
    )
    st.markdown(f"#### {title}")
    fig, ax = plt.subplots(figsize=(7.6, max(3.1, len(heatmap_df) * 0.72)))
    fig.patch.set_facecolor("#f5f9fb")
    ax.set_facecolor("#ffffff")
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".4f",
        cmap=dashboard_cmap,
        linewidths=1.2,
        linecolor="#ffffff",
        cbar=True,
        cbar_kws={"shrink": 0.72, "pad": 0.02},
        annot_kws={"fontsize": 9, "fontweight": "semibold", "color": "#163243"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=0, labelsize=9)
    ax.tick_params(axis="y", labelrotation=0, labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


def render_benchmark_tab() -> None:
    raw_results, result_path = load_benchmark_results()
    if not raw_results:
        st.warning("No benchmark artifact was found. Run the notebook training cells first.")
        return

    artifact_name = Path(result_path).name if result_path else "benchmark_results.pkl"
    st.markdown(
        f"""
        <div class="artifact-note">
            <strong>Loaded Benchmark Artifact</strong>
            <div>Source file: <code>{artifact_name}</code></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_results = extract_metric_results(raw_results)
    notebook_context = load_notebook_benchmark_context()
    notebook_context.update(benchmark_payload_context(raw_results))
    summary_df = build_legacy_summary_frame(metric_results)

    tuned_comparison_df = notebook_context.get("tuned_comparison", pd.DataFrame())
    if tuned_comparison_df.empty:
        tuned_comparison_df = summary_df.copy()

    stage_comparison_df = notebook_context.get("stage_comparison", pd.DataFrame())
    robust_cv_df = notebook_context.get("robust_cv", pd.DataFrame())
    baseline_comparison_df = notebook_context.get("baseline_comparison", pd.DataFrame())
    metadata_table = notebook_context.get("best_model_metadata_table", pd.DataFrame())
    selection_summary = notebook_context.get("selection_summary", "")
    best_model_name = parse_best_model_name(selection_summary, metadata_table)
    dashboard_winners_df = build_dashboard_winners_table(
        stage_comparison_df=stage_comparison_df,
        baseline_comparison_df=baseline_comparison_df,
        tuned_comparison_df=tuned_comparison_df,
    )

    if not dashboard_winners_df.empty:
        st.markdown("### Best Of 4 Models")
        render_styled_table(
            dashboard_winners_df,
            formatters={
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1 Score": "{:.4f}",
            },
        )

    st.markdown("### Tuned Model Summary")
    render_styled_table(
        tuned_comparison_df,
        formatters={
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "Best CV F1": "{:.4f}",
            "CV Mean": "{:.4f}",
            "CV Std": "{:.4f}",
        },
    )
    st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)
    render_metric_heatmap(
        tuned_comparison_df.rename(columns={"Best CV F1": "CV Mean"}),
        "Tuned Model Metric Heatmap",
    )

    if selection_summary:
        st.markdown("#### Notebook Selection Summary")
        st.code(selection_summary, language="text")
    if not metadata_table.empty:
        render_styled_table(metadata_table)

    if not stage_comparison_df.empty:
        st.markdown("### Baseline vs Tuned Comparison")
        render_styled_table(
            stage_comparison_df,
            formatters={
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1 Score": "{:.4f}",
            },
        )
        st.markdown("#### Baseline vs Tuned Model F1 Scores")
        fig, ax = plt.subplots(figsize=(8.2, 4.3))
        sns.barplot(
            data=stage_comparison_df,
            x="Model",
            y="F1 Score",
            hue="Stage",
            palette=NOTEBOOK_PRIMARY_COLORS,
            ax=ax,
        )
        ax.set_xlabel("Model")
        ax.set_ylabel("F1 Score")
        ax.set_ylim(0.94, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.legend(title="Stage", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    if not baseline_comparison_df.empty:
        st.markdown("### Baseline Model Results")
        render_styled_table(
            baseline_comparison_df,
            formatters={
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1 Score": "{:.4f}",
                "CV F1 Mean": "{:.4f}",
                "CV F1 Std": "{:.4f}",
            },
        )

    if not robust_cv_df.empty:
        st.markdown("### Robust Cross-Validation Check")
        render_styled_table(
            robust_cv_df,
            formatters={
                "Stratified CV Mean": "{:.4f}",
                "Stratified CV Std": "{:.4f}",
                "Repeated Stratified CV Mean": "{:.4f}",
                "Repeated Stratified CV Std": "{:.4f}",
                "Best Baseline Test F1": "{:.4f}",
            },
        )
        robust_plot_df = robust_cv_df.melt(
            id_vars=["Model"],
            value_vars=["Stratified CV Mean", "Repeated Stratified CV Mean", "Best Baseline Test F1"],
            var_name="Metric",
            value_name="Score",
        )
        st.markdown("#### Robust Validation Consistency")
        fig, ax = plt.subplots(figsize=(8.2, 4.3))
        sns.barplot(
            data=robust_plot_df,
            x="Model",
            y="Score",
            hue="Metric",
            palette=[NOTEBOOK_PRIMARY_COLORS[0], NOTEBOOK_PRIMARY_COLORS[1], "#88c2b0"],
            ax=ax,
        )
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_ylim(0.99, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.legend(frameon=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.markdown("### Final Tuned Model Evaluation")
    best_model_row = lookup_model_metrics(tuned_comparison_df, best_model_name)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Final Tuned Model", best_model_name)
    metric_cols[1].metric("Best CV F1", f"{float(best_model_row.get('Best CV F1', best_model_row.get('CV Mean', 0))):.4f}")
    metric_cols[2].metric("Test F1", f"{float(best_model_row.get('F1 Score', 0)):.4f}")
    metric_cols[3].metric("Search", str(best_model_row.get("Search Method", "Notebook")) or "Notebook")

    evaluation_figure = notebook_context.get("evaluation_figure_png", b"")
    if evaluation_figure:
        render_notebook_figure(evaluation_figure, "Notebook output: confusion matrix, ROC curve, and precision-recall curve")
        report_text = notebook_context.get("classification_report_text", "")
        if report_text:
            render_classification_report_block(report_text)
    else:
        best_model_file = resolve_best_benchmark_model_file()
        evaluation = notebook_context.get("best_model_evaluation", {})
        if not evaluation and best_model_file:
            evaluation = evaluate_best_benchmark_model(best_model_file)
        render_best_model_evaluation(evaluation, best_model_name)


def render_sidebar():
    available_models = list_available_pipeline_files()
    default_model = select_default_model_file()
    source_keys = list(NEWS_SOURCE_CATALOG.keys())

    if "selected_source_key" not in st.session_state:
        st.session_state.selected_source_key = None
    if "loaded_source_bundle" not in st.session_state:
        st.session_state.loaded_source_bundle = None
    if "fetch_min_chars" not in st.session_state:
        st.session_state.fetch_min_chars = DEFAULT_MIN_CHARS
    if "fetch_max_chars" not in st.session_state:
        st.session_state.fetch_max_chars = DEFAULT_MAX_CHARS
    if "selected_model_file" not in st.session_state:
        st.session_state.selected_model_file = None

    with st.sidebar:
        st.markdown("## Deployment Controls")
        if not available_models:
            st.error("No pipeline model artifacts were found in the `Models` folder.")
            return None, None, None

        selected_model = st.selectbox(
            "Model Artifact",
            options=[None] + available_models,
            index=([None] + available_models).index(st.session_state.selected_model_file)
            if st.session_state.selected_model_file in ([None] + available_models)
            else 0,
            format_func=lambda model_file: "Please select ..." if model_file is None else format_model_name(model_file),
        )
        previous_model_file = st.session_state.get("selected_model_file")
        if previous_model_file != selected_model:
            st.session_state.loaded_source_bundle = None
            clear_single_input()
            reset_sample_selection_state()
        st.session_state.selected_model_file = selected_model

        st.markdown("---")
        st.markdown("### Analyze URL")
        selected_source_key = st.selectbox(
            "News URL",
            options=[None] + source_keys,
            index=([None] + source_keys).index(st.session_state.selected_source_key)
            if st.session_state.selected_source_key in ([None] + source_keys)
            else 0,
            format_func=lambda key: "Please select ..." if key is None else NEWS_SOURCE_CATALOG[key]["label"],
        )
        previous_source_key = st.session_state.get("selected_source_key")
        if previous_source_key != selected_source_key:
            st.session_state.loaded_source_bundle = None
            clear_single_input()
            reset_sample_selection_state()
        st.session_state.selected_source_key = selected_source_key
        selected_source = NEWS_SOURCE_CATALOG[selected_source_key] if selected_source_key else None
        if selected_source:
            st.caption(selected_source["homepage"])
        min_chars = st.number_input(
            "Min article chars",
            min_value=10,
            max_value=5000,
            value=int(st.session_state.fetch_min_chars),
            step=10,
            disabled=selected_source_key is None,
        )
        max_chars = st.number_input(
            "Max article chars",
            min_value=100,
            max_value=50000,
            value=int(st.session_state.fetch_max_chars),
            step=100,
            disabled=selected_source_key is None,
        )
        st.session_state.fetch_min_chars = int(min_chars)
        st.session_state.fetch_max_chars = max(int(max_chars), int(min_chars) + 10)
        if selected_source_key is not None and st.session_state.fetch_max_chars != int(max_chars):
            st.caption(f"Max article chars adjusted to {st.session_state.fetch_max_chars} to stay above the minimum.")

        if st.button(
            "Load News Samples",
            type="primary",
            use_container_width=True,
            disabled=selected_source_key is None or selected_model is None,
        ):
            clear_single_input()
            reset_sample_selection_state()
            with st.spinner("Loading source samples..."):
                try:
                    st.session_state.loaded_source_bundle = fetch_source_bundle(
                        selected_source_key,
                        selected_model,
                        st.session_state.fetch_min_chars,
                        st.session_state.fetch_max_chars,
                    )
                except Exception as exc:
                    st.session_state.loaded_source_bundle = None
                    st.error(f"Failed to load news samples: {exc}")

        loaded_bundle = st.session_state.loaded_source_bundle
        current_loaded_bundle = (
            loaded_bundle
            if loaded_bundle
            and loaded_bundle.get("source_key") == selected_source_key
            and loaded_bundle.get("model_file") == selected_model
            else None
        )
        if current_loaded_bundle:
            st.caption(f"Loaded source: {current_loaded_bundle['source_label']}")
            st.caption(
                f"Real samples: {len(current_loaded_bundle['real_samples'])} | "
                f"Fake samples: {len(current_loaded_bundle['fake_samples'])}"
            )
            for note in current_loaded_bundle.get("load_notes", []):
                st.info(note)
            if len(current_loaded_bundle["real_samples"]) < REAL_SOURCE_SAMPLE_LIMIT:
                st.warning(
                    f"Only {len(current_loaded_bundle['real_samples'])} real samples could be loaded for this source."
                )

        if selected_model is not None or current_loaded_bundle is not None:
            st.markdown("---")
            metadata = load_best_model_metadata()
            metadata_rows = []
            if selected_model is not None:
                metadata_rows.append(("Selected model", format_model_name(selected_model)))
            if current_loaded_bundle is not None:
                metadata_rows.append(("Loaded news source", current_loaded_bundle["source_label"]))
                metadata_rows.append(("Min article chars", str(st.session_state.fetch_min_chars)))
                metadata_rows.append(("Max article chars", str(st.session_state.fetch_max_chars)))
            metadata_rows.extend(
                [
                    ("Best notebook model", metadata.get("best_model_name", "Not available")),
                    ("Best notebook slug", metadata.get("best_model_slug", "Not available")),
                ]
            )
            st.markdown("#### Deployment Metadata")
            render_styled_table(pd.DataFrame(metadata_rows, columns=["Item", "Value"]))

    return selected_model, st.session_state.fetch_min_chars, st.session_state.fetch_max_chars


def clear_single_input() -> None:
    st.session_state.single_input = ""


def reset_sample_selection_state() -> None:
    keys_to_clear = []
    for key in st.session_state.keys():
        if (
            "_sample_select" in key
            or key in {"loaded_sample_group"}
            or key.endswith("_selection_signature")
        ):
            keys_to_clear.append(key)

    for key in keys_to_clear:
        st.session_state.pop(key, None)


def format_sample_label(sample: dict, index: int) -> str:
    title = sample.get("title", "Untitled sample").strip() or "Untitled sample"
    truncated_title = title if len(title) <= 72 else f"{title[:69]}..."
    return f"{index + 1}. {truncated_title}"


def sync_selected_sample_to_input(
    samples: list[dict],
    selector_key: str,
    source_signature: str,
    selected_value,
) -> None:
    if not samples or selected_value is None:
        return

    selected_index = int(selected_value)
    if selected_index >= len(samples):
        return

    signature = f"{source_signature}:{selected_index}"
    tracking_key = f"{selector_key}_selection_signature"
    previous_signature = st.session_state.get(tracking_key)
    st.session_state[tracking_key] = signature
    if previous_signature == signature:
        return

    st.session_state.single_input = samples[selected_index]["text"]


def render_sample_selector(
    samples: list[dict],
    tab_label: str,
    empty_message: str,
    source_signature: str,
    source_caption: str,
    placeholder_label: str,
) -> None:
    if not samples:
        st.markdown(f'<div class="plain-alert">{empty_message}</div>', unsafe_allow_html=True)
        return

    selector_key = f"{tab_label.lower().replace(' ', '_')}_{source_signature}_sample_select"
    selected_value = st.selectbox(
        f"{tab_label} dropdown",
        options=[None] + list(range(len(samples))),
        format_func=lambda value: placeholder_label if value is None else format_sample_label(samples[value], value),
        key=selector_key,
    )
    if selected_value is None:
        return

    sync_selected_sample_to_input(samples, selector_key, source_signature, selected_value)
    selected_sample = samples[int(selected_value)]
    st.caption(selected_sample.get("published", "") or source_caption)
    if selected_sample.get("url"):
        st.caption(selected_sample["url"])


def render_predict_tab(selected_model: str, min_chars: int, max_chars: int):
    if "single_input" not in st.session_state:
        st.session_state.single_input = ""
    raw_loaded_bundle = st.session_state.get("loaded_source_bundle")
    loaded_bundle = (
        raw_loaded_bundle
        if raw_loaded_bundle and raw_loaded_bundle.get("model_file") == selected_model
        else None
    )
    input_mode = st.radio(
        "Choose how to provide the article",
        options=["Use loaded samples", "Paste manually"],
        key="predict_input_mode",
        horizontal=True,
    )
    previous_input_mode = st.session_state.get("predict_input_mode_signature")
    if previous_input_mode is None:
        st.session_state.predict_input_mode_signature = input_mode
    elif previous_input_mode != input_mode:
        clear_single_input()
        reset_sample_selection_state()
        st.session_state.predict_input_mode_signature = input_mode

    if loaded_bundle:
        st.markdown(
            f"""
            <div class="info-panel">
                <strong>Loaded sample source:</strong> {loaded_bundle['source_label']}<br>
                Real articles fetched: {len(loaded_bundle['real_samples'])} / {REAL_SOURCE_SAMPLE_LIMIT}<br>
                Predict-tab fake samples: {len(loaded_bundle['fake_samples'])}
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif input_mode == "Use loaded samples":
        st.markdown(
            '<div class="soft-note">Load samples from the sidebar first, or switch to <strong>Paste manually</strong> and enter article text directly.</div>',
            unsafe_allow_html=True,
        )

    if input_mode == "Use loaded samples":
        sample_group = st.radio(
            "Choose loaded sample type",
            options=["Real News Samples", "Fake News Samples"],
            key="loaded_sample_group",
            horizontal=True,
        )
        previous_sample_group = st.session_state.get("loaded_sample_group_signature")
        if previous_sample_group is None:
            st.session_state.loaded_sample_group_signature = sample_group
        elif previous_sample_group != sample_group:
            clear_single_input()
            reset_sample_selection_state()
            st.session_state.loaded_sample_group_signature = sample_group

        if sample_group == "Real News Samples":
            render_sample_selector(
                samples=loaded_bundle["real_samples"] if loaded_bundle else [],
                tab_label="Real News Sample",
                empty_message="No real news samples are loaded yet.",
                source_signature=loaded_bundle["source_key"] if loaded_bundle else "real",
                source_caption=loaded_bundle["source_label"] if loaded_bundle else "Real source",
                placeholder_label="Please select Real sample",
            )
        else:
            render_sample_selector(
                samples=loaded_bundle["fake_samples"] if loaded_bundle else [],
                tab_label="Fake News Sample",
                empty_message="No fake news samples are loaded yet.",
                source_signature=f"{loaded_bundle['source_key']}_fake" if loaded_bundle else "fake",
                source_caption="Local fake-news dataset",
                placeholder_label="Please select Fake sample",
            )
    else:
        st.caption("Manual mode is active. Paste fake or real news directly into the text box below.")

    if selected_model is None:
        st.markdown(
            '<div class="soft-note">Please select a <strong>Model Artifact</strong> before analyzing article text.</div>',
            unsafe_allow_html=True,
        )

    text = st.text_area(
        "Article Text",
        key="single_input",
        height=250,
        placeholder="Paste fake or real news here, or switch to `Use loaded samples` to auto-load text from the selected source...",
    )

    action_col_1, action_col_2, _action_spacer = st.columns([1, 1, 2])
    analyze_clicked = action_col_1.button(
        "Analyze Article",
        type="primary",
        use_container_width=True,
        disabled=selected_model is None,
    )
    action_col_2.button("Clear", on_click=clear_single_input, use_container_width=True)

    if analyze_clicked:
        is_valid, message = validate_text_input(text, min_chars, max_chars)
        if not is_valid:
            st.warning(message)
            return

        with st.spinner("Running synchronized professor-aligned prediction..."):
            prediction = predict_news(text, DEFAULT_THRESHOLD, selected_model)
        render_result_card(prediction)
        render_preprocessing_trace(prediction["trace"])


def main():
    st.markdown('<div class="hero-title">News Classification</div>', unsafe_allow_html=True)
    

    selected_model, min_chars, max_chars = render_sidebar()

    tab_predict, tab_benchmark, tab_pipeline = st.tabs(
        ["Predict", "Dashboard", "News Classification Pipeline"]
    )

    with tab_predict:
        render_predict_tab(selected_model, min_chars, max_chars)
    with tab_benchmark:
        render_benchmark_tab()
    with tab_pipeline:
        render_pipeline_tab()


if __name__ == "__main__":
    main()

import os
import re
import pickle
import string
from pathlib import Path

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
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

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

PRIMARY_COLORS = ["#3CB371", "#0f6096"]
FAKE_COLOR, REAL_COLOR = PRIMARY_COLORS
LABEL_MAP = {0: "Fake News", 1: "Real News"}


def ensure_nltk_resource(resource_path: str, download_name: str) -> None:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name, quiet=True)


for resource_path, download_name in [
    ("tokenizers/punkt", "punkt"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
]:
    ensure_nltk_resource(resource_path, download_name)


st.set_page_config(
    page_title="Professor Pipeline News Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Source+Sans+3:wght@400;500;600;700&display=swap');

    :root {
        --ink: #102028;
        --sea: #0f6096;
        --mint: #3CB371;
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
        border-left: 8px solid var(--sea);
        border-radius: 16px;
        padding: 1rem 1.15rem;
        box-shadow: 0 10px 28px rgba(16, 32, 40, 0.08);
        margin-bottom: 1rem;
    }

    .result-card.fake {
        border-left-color: var(--mint);
    }

    .result-card.real {
        border-left-color: var(--sea);
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
        height: 8px;
        border-radius: 999px;
        overflow: hidden;
    }

    .confidence-fill {
        height: 8px;
        border-radius: 999px;
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


def preprocess_with_trace(text: str) -> dict:
    normalized_text = normalize_text(text)
    clean_text = clean_text_content(normalized_text)
    tokens = tokenize_text(clean_text)
    filtered_tokens = remove_stopwords(tokens)
    lemmatized_tokens = lemmatize_with_pos(filtered_tokens)
    cleaned_text = " ".join(lemmatized_tokens)
    return {
        "raw_text": text,
        "normalized_text": normalized_text,
        "clean_text": clean_text,
        "tokens": tokens,
        "tokens_no_stopwords": filtered_tokens,
        "lemmatized_tokens": lemmatized_tokens,
        "cleaned_text": cleaned_text,
    }


def validate_text_input(text: str, min_chars: int, max_chars: int) -> tuple[bool, str]:
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


def extract_text_from_url(url: str, extraction_strategy: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        )
    }

    if extraction_strategy == "Trafilatura" and trafilatura is not None:
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
                <div class="confidence-fill" style="width:{confidence_pct:.1f}%; background:{fill_color};"></div>
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
    st.dataframe(probability_df.style.format({"Probability": "{:.4f}"}), hide_index=True, use_container_width=True)


def render_preprocessing_trace(trace: dict) -> None:
    with st.expander("Show Professor-Aligned Preprocessing Trace", expanded=False):
        stage_rows = [
            ("Text Normalization", trace["normalized_text"][:500]),
            ("Text Cleaning", trace["clean_text"][:500]),
            ("Tokenization", ", ".join(trace["tokens"][:40])),
            ("Remove Stopwords", ", ".join(trace["tokens_no_stopwords"][:40])),
            ("POS-Aware Lemmatization", ", ".join(trace["lemmatized_tokens"][:40])),
            ("Final Modeling Text", trace["cleaned_text"][:500]),
        ]
        stage_df = pd.DataFrame(stage_rows, columns=["Pipeline Step", "Preview"])
        st.dataframe(stage_df, hide_index=True, use_container_width=True)
        st.caption(
            f"Token counts: raw tokens={len(trace['tokens'])}, after stopword removal={len(trace['tokens_no_stopwords'])}, lemmas={len(trace['lemmatized_tokens'])}"
        )


def render_pipeline_tab() -> None:
    st.markdown("### Professor-Aligned Pipeline")
    stages = [
        ("01", "Problem and Dataset", "Acquire the fake-news dataset and define the binary classification task."),
        ("02", "Data Understanding", "Inspect structure, class balance, subject mix, and exploratory plots."),
        ("03", "Data Cleaning", "Remove duplicates, standardize columns, and assemble the raw modeling text."),
        ("04", "Text Preprocessing", "Normalize, clean, tokenize, remove stopwords, and lemmatize with POS tagging."),
        ("05", "Data Split", "Perform stratified train/test split."),
        ("06", "Word Vectors", "Transform cleaned text into TF-IDF vectors."),
        ("07", "ML Algorithm Building", "Train baseline Logistic Regression, Naive Bayes, Linear SVM, and Random Forest."),
        ("08", "Hyperparameter Tuning", "Run GridSearchCV and RandomizedSearchCV refinement."),
        ("09", "Tuned Model", "Select the strongest tuned candidate using validation F1."),
        ("10", "Evaluation", "Inspect confusion matrix, ROC, PR curve, and benchmark comparisons."),
        ("11", "Prediction", "Use the final synchronized pipeline for live inference."),
    ]

    columns = st.columns(2)
    for idx, (number, title, description) in enumerate(stages):
        with columns[idx % 2]:
            st.markdown(
                f"""
                <div class="pipeline-card">
                    <div class="pipeline-step"><strong>Stage {number}</strong></div>
                    <div class="pipeline-step"><strong>{title}</strong></div>
                    <div class="pipeline-step">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_benchmark_tab() -> None:
    results, result_path = load_benchmark_results()
    if not results:
        st.warning("No benchmark artifact was found. Run the notebook training cells first.")
        return

    st.caption(f"Loaded benchmark artifact: `{result_path}`")

    rows = []
    for model_name, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "Model": model_name,
                "Stage": metrics.get("stage", ""),
                "Search Method": metrics.get("search_method", ""),
                "Accuracy": float(metrics.get("accuracy", 0)),
                "Precision": float(metrics.get("precision", 0)),
                "Recall": float(metrics.get("recall", 0)),
                "F1": float(metrics.get("f1", 0)),
                "CV Mean": float(metrics.get("cv_mean", 0)),
            }
        )

    benchmark_df = pd.DataFrame(rows).sort_values(["CV Mean", "F1"], ascending=False)
    st.dataframe(benchmark_df, hide_index=True, use_container_width=True)

    if benchmark_df.empty:
        return

    best_row = benchmark_df.iloc[0]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Best Model", best_row["Model"])
    metric_cols[1].metric("Search", best_row["Search Method"] or "N/A")
    metric_cols[2].metric("F1", f"{best_row['F1']:.4f}")
    metric_cols[3].metric("CV Mean", f"{best_row['CV Mean']:.4f}")

    chart_df = benchmark_df.melt(
        id_vars=["Model", "Stage", "Search Method"],
        value_vars=["Accuracy", "Precision", "Recall", "F1", "CV Mean"],
        var_name="Metric",
        value_name="Score",
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=chart_df, x="Metric", y="Score", hue="Model", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Stored Notebook Benchmark Comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    st.pyplot(fig, use_container_width=True)


def render_sidebar():
    available_models = list_available_pipeline_files()
    default_model = select_default_model_file()

    with st.sidebar:
        st.markdown("## Deployment Controls")
        if not available_models:
            st.error("No pipeline model artifacts were found in the `Models` folder.")
            return None, None, None, None

        selected_model = st.selectbox(
            "Model Artifact",
            options=available_models,
            index=available_models.index(default_model) if default_model in available_models else 0,
            format_func=format_model_name,
        )
        threshold = st.slider("Real-news threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05)
        min_chars = st.number_input("Minimum characters", min_value=10, max_value=500, value=30, step=10)
        max_chars = st.number_input("Maximum characters", min_value=1000, max_value=100000, value=20000, step=1000)
        extraction_strategy = st.selectbox("URL extraction strategy", ["Trafilatura", "BeautifulSoup"])

        st.markdown("---")
        metadata = load_best_model_metadata()
        metadata_rows = [
            ("Selected model", format_model_name(selected_model)),
            ("Best notebook model", metadata.get("best_model_name", "Not available")),
            ("Best notebook slug", metadata.get("best_model_slug", "Not available")),
            ("Artifact folder", str(ARTIFACT_DIR) if ARTIFACT_DIR.exists() else "Models folder not found"),
        ]
        st.dataframe(pd.DataFrame(metadata_rows, columns=["Item", "Value"]), hide_index=True, use_container_width=True)

    return selected_model, float(threshold), int(min_chars), int(max_chars), extraction_strategy


REAL_SAMPLE = """Reuters reports that major government agencies have expanded public health support measures after a parliamentary vote, with officials saying the policy will reduce barriers to treatment and improve emergency intervention access."""

FAKE_SAMPLE = """BREAKING: secret global elites have activated invisible towers that control citizens through microchips hidden in medicine, according to anonymous insiders who say mainstream media is covering it up."""


def render_predict_tab(selected_model: str, threshold: float, min_chars: int, max_chars: int):
    if "single_input" not in st.session_state:
        st.session_state.single_input = ""

    col1, col2, col3 = st.columns([1, 1, 1])
    if col1.button("Load Real Sample"):
        st.session_state.single_input = REAL_SAMPLE
    if col2.button("Load Fake Sample"):
        st.session_state.single_input = FAKE_SAMPLE
    if col3.button("Clear"):
        st.session_state.single_input = ""

    text = st.text_area(
        "Article Text",
        key="single_input",
        height=250,
        placeholder="Paste a full article or a long excerpt for more reliable classification...",
    )

    if st.button("Analyze Article", type="primary"):
        is_valid, message = validate_text_input(text, min_chars, max_chars)
        if not is_valid:
            st.warning(message)
            return

        with st.spinner("Running synchronized professor-aligned prediction..."):
            prediction = predict_news(text, threshold, selected_model)
        render_result_card(prediction)
        render_preprocessing_trace(prediction["trace"])


def render_url_tab(selected_model: str, threshold: float, min_chars: int, max_chars: int, extraction_strategy: str):
    if "url_input" not in st.session_state:
        st.session_state.url_input = ""

    url = st.text_input(
        "News URL",
        key="url_input",
        placeholder="https://example.com/news/article",
    )

    if st.button("Fetch and Analyze URL", type="primary"):
        if not url.strip().startswith(("http://", "https://")):
            st.warning("Please enter a valid http/https URL.")
            return

        with st.spinner("Fetching and extracting article text..."):
            extracted_text = extract_text_from_url(url, extraction_strategy)

        is_valid, message = validate_text_input(extracted_text, min_chars, max_chars)
        if not is_valid:
            st.warning(message)
            st.text_area("Extracted Preview", value=extracted_text[:3000], height=220)
            return

        st.text_area("Extracted Preview", value=extracted_text[:3000], height=220)
        prediction = predict_news(extracted_text, threshold, selected_model)
        render_result_card(prediction)
        render_preprocessing_trace(prediction["trace"])


def main():
    st.markdown('<div class="hero-title">Professor-Aligned News Authenticity Classifier</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">This deployment app mirrors the notebook pipeline: structural cleaning, text normalization, text cleaning, tokenization, stopword removal, POS-aware lemmatization, TF-IDF vectorization, tuned model selection, and final prediction.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-panel">Deployment is synchronized to the enhanced notebook flow. The app reads pipeline artifacts only from the <code>Models</code> folder, uses the same preprocessing steps as the notebook, and reads the saved benchmark metadata when available.</div>',
        unsafe_allow_html=True,
    )

    selected_model, threshold, min_chars, max_chars, extraction_strategy = render_sidebar()
    if not selected_model:
        return

    tab_predict, tab_url, tab_benchmark, tab_pipeline = st.tabs(
        ["Predict", "Analyze URL", "Benchmark", "Pipeline"]
    )

    with tab_predict:
        render_predict_tab(selected_model, threshold, min_chars, max_chars)
    with tab_url:
        render_url_tab(selected_model, threshold, min_chars, max_chars, extraction_strategy)
    with tab_benchmark:
        render_benchmark_tab()
    with tab_pipeline:
        render_pipeline_tab()


if __name__ == "__main__":
    main()

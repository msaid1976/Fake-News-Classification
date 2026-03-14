# Library Imports
import math
import os
import pickle
import streamlit as st  # To create the application interface
import joblib   # To load models and vectors
import re   # To use regular expressions in text processing
import nltk # Natural Language Toolkit for text processing
import pandas as pd  # For benchmark tables
import requests  # To fetch live URL content
import matplotlib.pyplot as plt  # For benchmark charts
import seaborn as sns  # For benchmark charts
from bs4 import BeautifulSoup  # To extract article content from URL
from nltk.corpus import stopwords   # To access stopwords
from nltk.stem import PorterStemmer # For word stemming
from nltk.tokenize import word_tokenize # To convert words into tokens
import numpy as np  # For numerical operations

# Download NLTK data
# Check if NLTK tokenizer data is avialable, download if not found
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
# Check if NLTK stopwords data is avialable, download if not found
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# This is for url extraction using trafilatura, which is more precise than BeautifulSoup but can be less reliable in some environments.
try:
    import trafilatura
except Exception:
    trafilatura = None

# Set up page configuration
st.set_page_config(
    page_title="News Classification",   # Title shown in browser tab
    page_icon="",   # Set no icon
    layout="wide",  # Set layout
    initial_sidebar_state="expanded"    
)

# Custom CSS for styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Source+Sans+3:wght@400;500;600;700&display=swap');

    :root {
        --ink: #102028;
        --sky: #e8f1f8;
        --sand: #f7f3ea;
        --accent: #0b7a75;
        --warn: #af3b2f;
    }

    .stApp {
        font-family: 'Source Sans 3', sans-serif;
        background: #f8fafc;
        color: var(--ink);
    }

    .main-header {
        font-family: 'Fraunces', serif;
        font-size: 3rem;
        line-height: 1.1;
        margin-bottom: 0.8rem;
        color: #17303a;
        letter-spacing: 0.01em;
        text-align: center;
    }

    .hero-title {
        font-family: 'Fraunces', serif;
        font-size: 2.2rem;
        line-height: 1.1;
        margin-bottom: 0.3rem;
        color: #17303a;
        letter-spacing: 0.01em;
    }

    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.88;
        margin-bottom: 1.2rem;
    }

    .sub-header {
        font-size: 1.35rem;
        color: #334a55;
        margin-bottom: 1rem;
    }

    .info-box {
        background-color: var(--sky);
        padding: 16px;
        border-radius: 12px;
        border-left: 5px solid var(--accent);
        margin-bottom: 20px;
        color: #17303a;
        box-shadow: 0 8px 20px rgba(19, 43, 58, 0.08);
    }

    .stTabs [role="tab"] {
        color: #41545d !important;
        font-weight: 700;
    }

    .stTabs [role="tab"][aria-selected="true"] {
        color: var(--accent) !important;
    }

    .stTextInput label,
    .stTextArea label,
    .stSelectbox label,
    .stNumberInput label {
        color: #334a55 !important;
        font-weight: 700 !important;
    }

    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stTextArea label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSlider label {
        color: #f3f6fb !important;
        -webkit-text-fill-color: #f3f6fb !important;
    }

    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] .stCaption p,
    [data-testid="stSidebar"] small {
        color: rgba(243, 246, 251, 0.72) !important;
        -webkit-text-fill-color: rgba(243, 246, 251, 0.72) !important;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div,
    div[data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.92) !important;
        border: 1px solid rgba(19, 43, 58, 0.20) !important;
        border-radius: 12px !important;
    }

    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea,
    div[data-baseweb="select"] * {
        color: #17303a !important;
        -webkit-text-fill-color: #17303a !important;
    }

    div[data-baseweb="input"] input::placeholder,
    div[data-baseweb="textarea"] textarea::placeholder {
        color: #6a7b86 !important;
        -webkit-text-fill-color: #6a7b86 !important;
        opacity: 1 !important;
    }

    .stNumberInput button {
        background: rgba(255, 255, 255, 0.92) !important;
        border: 1px solid rgba(19, 43, 58, 0.20) !important;
        color: #17303a !important;
    }

    .stNumberInput button svg,
    .stNumberInput button span,
    .stNumberInput button p {
        fill: #17303a !important;
        color: #17303a !important;
        -webkit-text-fill-color: #17303a !important;
    }

    .stNumberInput button:hover {
        background: #eef6ff !important;
    }

    div[data-testid="stExpander"] details > summary {
        background: rgba(255, 255, 255, 0.92) !important;
        color: #17303a !important;
        border: 1px solid rgba(19, 43, 58, 0.20) !important;
        border-radius: 12px !important;
    }

    div[data-testid="stExpander"] details > summary p,
    div[data-testid="stExpander"] details > summary span,
    div[data-testid="stExpander"] details > summary svg {
        color: #17303a !important;
        fill: #17303a !important;
    }

    div[data-testid="stExpander"] details[open] > summary {
        border-bottom-left-radius: 0 !important;
        border-bottom-right-radius: 0 !important;
    }

    div[data-testid="stExpander"] details > div {
        background: rgba(255, 255, 255, 0.82) !important;
        border: 1px solid rgba(19, 43, 58, 0.12) !important;
        border-top: none !important;
        border-bottom-left-radius: 12px !important;
        border-bottom-right-radius: 12px !important;
        padding-top: 0.65rem !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #162841 0%, #1e3454 100%);
        color: #f6f9ff !important;
        border: 1px solid rgba(200, 216, 236, 0.30);
        border-radius: 12px;
        font-weight: 700;
        min-height: 2.85rem;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 10px rgba(19, 31, 49, 0.18);
    }

    .stButton > button:hover {
        color: #ffffff !important;
        border-color: rgba(224, 238, 255, 0.55);
        transform: translateY(-1px);
        box-shadow: 0 8px 16px rgba(19, 31, 49, 0.24);
    }

    .stButton > button:focus {
        color: #ffffff !important;
        outline: none;
        box-shadow: 0 0 0 0.2rem rgba(11, 122, 117, 0.35);
    }

    .stButton > button:disabled {
        background: #b7c0c7;
        color: #36464f !important;
        border-color: #9ca8b0;
        box-shadow: none;
        cursor: not-allowed;
    }

    .stButton > button span,
    .stButton > button p,
    .stButton > button div {
        color: #f6f9ff !important;
        -webkit-text-fill-color: #f6f9ff !important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stWidgetLabel"] {
        color: #17303a !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(19, 43, 58, 0.12);
        border-radius: 12px;
        padding: 0.65rem 0.9rem;
        box-shadow: 0 4px 12px rgba(19, 43, 58, 0.08);
    }

    div[data-testid="stMetric"] label {
        color: #334a55 !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #132b39 !important;
        font-weight: 800 !important;
        letter-spacing: 0.01em;
    }

    div[data-testid="stMetricDelta"] {
        color: #3f5763 !important;
    }

    .model-card {
        background: rgba(11, 122, 117, 0.08);
        border: 1px solid rgba(11, 122, 117, 0.28);
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 4px;
    }

    .model-card-name {
        font-weight: 700;
        font-size: 0.95rem;
        color: #0b7a75;
        word-break: break-all;
        font-family: 'Source Sans 3', monospace;
    }

    .result-card {
        border-radius: 14px;
        padding: 16px 18px;
        margin-top: 12px;
        border: 1px solid rgba(0, 0, 0, 0.08);
        box-shadow: 0 8px 22px rgba(18, 41, 52, 0.12);
        background: #ffffff;
    }

    .result-card.news-real {
        border-left: 8px solid #0b7a75;
        background: linear-gradient(120deg, #ffffff 0%, #e9f8f6 100%);
    }

    .result-card.news-fake {
        border-left: 8px solid #af3b2f;
        background: linear-gradient(120deg, #ffffff 0%, #fdece8 100%);
    }

    .result-label {
        font-family: 'Fraunces', serif;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .result-confidence {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .conf-bar-track {
        background: #e2e8f0;
        border-radius: 3px;
        height: 6px;
        margin-bottom: 0.6rem;
        overflow: hidden;
    }

    .conf-bar {
        height: 6px;
        border-radius: 3px;
    }

    .conf-bar.news-real {
        background: #0b7a75;
    }

    .conf-bar.news-fake {
        background: #af3b2f;
    }

    .result-note {
        font-size: 0.9rem;
        opacity: 0.82;
    }

    .open-link-wrap {
        display: flex;
        align-items: stretch;
        margin: 0;
        padding: 0;
    }

    .open-link-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 2.85rem;
        text-decoration: none !important;
        background: rgba(255, 255, 255, 0.90);
        color: #17303a !important;
        border: 1px solid rgba(19, 43, 58, 0.22);
        border-radius: 12px;
        font-weight: 700;
        line-height: 1.15;
        padding: 0.5rem 1rem;
        transition: all 0.18s ease;
        box-shadow: 0 2px 8px rgba(19, 43, 58, 0.08);
    }

    .open-link-btn:hover {
        color: #102028 !important;
        border-color: rgba(19, 43, 58, 0.38);
        transform: translateY(-1px);
        box-shadow: 0 6px 12px rgba(19, 43, 58, 0.13);
    }

    .open-link-btn.disabled {
        color: #7b8b95 !important;
        border-color: rgba(122, 141, 154, 0.28);
        background: rgba(255, 255, 255, 0.72);
        box-shadow: none;
        pointer-events: none;
        user-select: none;
    }
    </style>
    """, unsafe_allow_html=True) # To allow HTML rendering in streamlit

# Load the model with caching to improve performance
@st.cache_resource  # Decorator to cache the resource and avoid reloading on every interaction
def load_model():
    """
    Load the pre-trained model from pkl file
    
    The function attempts to load a trained model saved as a pickel file
    It uses caching to improve performance by avoiding reloading on each interaction
    
    Returns:
        model: The loaded machine learning model object
        None: If the model file is not found or an error occurs
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
        Exception: For any other errors during model Loading
    """
    try:
        # Prefer pipeline artifacts because they keep the vectorizer and classifier in sync
        model_files = [
            "linear_svm_pipeline.pkl",
            "naive_bayes_pipeline.pkl",
            "logistic_regression_pipeline.pkl",
            "random_forest_pipeline.pkl",
            "linear_svm_model.pkl",
            "naive_bayes_model.pkl",
            "logistic_regression_model.pkl",
            "random_forest_model.pkl"
        ]

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "Models")
        for model_file in model_files:
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                return model, model_file

        raise FileNotFoundError
    except FileNotFoundError:
        st.error("No trained model file was found. Please make sure the model .pkl files are in the Models folder.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {str(e)}")
        return None, None

def get_available_model_options():
    """
    Return available pipeline model files and display labels
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "Models")
    model_files = [
        "random_forest_pipeline.pkl",
        "linear_svm_pipeline.pkl",
        "logistic_regression_pipeline.pkl",
        "naive_bayes_pipeline.pkl",
    ]

    options = []
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        if os.path.exists(model_path):
            options.append((model_file, format_model_name(model_file)))
    return options

def get_best_model_file():
    """
    Pick the best available model file using saved benchmark F1 scores
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "Models")
    result_path = os.path.join(model_dir, "model_RF_results.pkl")
    model_name_to_file = {
        "Random Forest": "random_forest_pipeline.pkl",
        "Linear SVM": "linear_svm_pipeline.pkl",
        "Logistic Regression": "logistic_regression_pipeline.pkl",
        "Naive Bayes": "naive_bayes_pipeline.pkl",
    }

    if os.path.exists(result_path):
        with open(result_path, "rb") as file:
            results = pickle.load(file)

        best_name = None
        best_f1 = -1.0
        for model_name, metrics in results.items():
            if isinstance(metrics, dict) and "f1" in metrics:
                f1_score = float(metrics["f1"])
                if f1_score > best_f1 and model_name in model_name_to_file:
                    best_f1 = f1_score
                    best_name = model_name

        if best_name:
            return model_name_to_file[best_name]

    options = get_available_model_options()
    return options[0][0] if options else None

@st.cache_resource
def load_selected_model(model_file):
    """
    Load a specific saved model artifact
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "Models")
    model_path = os.path.join(model_dir, model_file)
    model = joblib.load(model_path)
    return model, model_file

# Try to load TF-IDF vectorizer if it exists
@st.cache_resource  # Decorator to cache the resources
def load_vectorizer():
    """
    Load the TF-IDF vectorizer froma file or create a basic one if not found
    
    This function attemps to load a pre-fitted TF-IDF vectorizer, if the file is not found, it creates a basic vectorizer as a fallback option
    
    Returns:
        vectorizer: The loaded or newly created TF-IDF vectorizer object
    """
    try:
        # Check if a vectorizer file exists
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "Models")
        vectorizer = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
        return vectorizer
    except:
        st.warning("TF-IDF vectorizer not found. Creating a basic one. For accurate results, please ensure you have the correct vectorizer.")
        # Create a basic vectorizer as fallback
        from sklearn.feature_extraction.text import TfidfVectorizer
        return TfidfVectorizer(max_features=5000)

# Text preprocessing function
def preprocess_text(text):
    """
    Preprocess and clean text data before passing to the model
    
    This function performs NLP preprocessing steps:
    1. Converts text to lowercase
    2. Removes special characters and digits
    3. Tokenizes the text into words
    4. Removes stopwords and short words
    5. Applies stemming to reduce words to their root form
    
    Args:
        text(str): The raw text input to be processed
    
    Returns:
        str: The cleaned and processed text as a single string
    """
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)

def predict_news(text, threshold=0.50, selected_model_file=None):
    """
    Predict whether a news ariticle is real or fake
    
    This function takes raw text input, preprocesses it, vectorizes it, and uses the trained model to make a prediction about its authenticity.
    
    Args:
        text(str): The news text to be analyzed
        
    Returns:
        tuple: A tuple containing:
                - result (str): "Real News" or "Fake News"
                - confidence (float): Confidence score for the prediction
                - probabilities (array): Array of probablilities for both classes
        
    The model is a binary classification model
    0 = Fake News, 1 = Real News
    """
    # Preprocess the text
    cleaned_text = preprocess_text(text)
    
    # Load Model
    if selected_model_file:
        model, model_name = load_selected_model(selected_model_file)
    else:
        model, model_name = load_model()
    if model is None:
        raise FileNotFoundError("No trained model is available.")

    # Pipelines already include the matching vectorizer
    if hasattr(model, "steps"):
        model_input = [cleaned_text]
    else:
        vectorizer = load_vectorizer()
        model_input = vectorizer.transform([cleaned_text])

    # Make prediction
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(model_input)[0]
    elif hasattr(model, "decision_function"):
        decision_score = float(model.decision_function(model_input)[0])
        real_probability = 1 / (1 + math.exp(-decision_score))
        probability = np.array([1 - real_probability, real_probability])
    else:
        prediction = model.predict(model_input)[0]
        probability = np.array([1.0, 0.0]) if prediction == 0 else np.array([0.0, 1.0])
    
    # Prediction Results
    real_probability = float(probability[1])
    prediction = 1 if real_probability >= threshold else 0
    result = "Real News" if prediction == 1 else "Fake News"
    confidence = real_probability if prediction == 1 else 1 - real_probability
    
    return result, confidence, probability, model_name, cleaned_text

def validate_text_input(text, min_chars=30, max_chars=20000):
    """
    Validate article text before passing it to the model
    """
    clean_text = text.strip()
    if not clean_text:
        return False, "Please enter article text before running prediction."
    if len(clean_text) < min_chars:
        return False, f"Article text is too short. Minimum length is {min_chars} characters."
    if len(clean_text) > max_chars:
        return False, f"Article text is too long. Maximum length is {max_chars} characters."
    return True, ""

def render_prediction_card(result, confidence, probabilities, model_name):
    """
    Render prediction results using the frontend-style card
    """
    badge_class = "news-real" if result == "Real News" else "news-fake"
    confidence_pct = confidence * 100

    st.markdown(
        f"""
        <div class="result-card {badge_class}">
            <div class="result-label">{result}</div>
            <div class="result-confidence">Confidence: {confidence_pct:.2f}%</div>
            <div class="conf-bar-track">
                <div class="conf-bar {badge_class}" style="width:{confidence_pct:.1f}%"></div>
            </div>
            <div class="result-note">Loaded model: {model_name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_preprocessing_expander(cleaned_text):
    """
    Show the preprocessed text in a collapsed expander
    """
    with st.expander("Show Preprocessed Text", expanded=False):
        st.text_area(
            "Cleaned text",
            value=cleaned_text,
            height=160,
            disabled=True,
        )

def render_open_link(label, url, disabled=False):
    """
    Render a styled external link button
    """
    safe_label = label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    if disabled:
        st.markdown(
            f'<div class="open-link-wrap"><span class="open-link-btn disabled">{safe_label}</span></div>',
            unsafe_allow_html=True,
        )
        return

    safe_url = url.replace("&", "&amp;").replace('"', "&quot;")
    st.markdown(
        (
            '<div class="open-link-wrap">'
            f'<a class="open-link-btn" href="{safe_url}" target="_blank" '
            f'rel="noopener noreferrer">{safe_label}</a>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )

def load_benchmark_results():
    """
    Load the stored benchmark results produced by the notebook training flow
    """
    result_files = [
        "model_RF_results.pkl",
        "model_LSVM_results.pkl",
        "model_metrics.pkl",
    ]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "Models")
    for result_file in result_files:
        result_path = os.path.join(model_dir, result_file)
        if os.path.exists(result_path):
            with open(result_path, "rb") as file:
                return pickle.load(file), result_file

    raise FileNotFoundError("No benchmark result file was found.")

def load_sidebar_metadata(model_name):
    """
    Collect lightweight metadata for the sidebar footer
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "Models")
    vectorizer_name = "tfidf_vectorizer.pkl" if os.path.exists(os.path.join(model_dir, "tfidf_vectorizer.pkl")) else "missing"

    benchmark_name = "missing"
    for result_file in ["model_RF_results.pkl", "model_LSVM_results.pkl", "model_metrics.pkl"]:
        if os.path.exists(os.path.join(model_dir, result_file)):
            benchmark_name = result_file
            break

    return {
        "Predictor": model_name,
        "Vectorizer": vectorizer_name,
        "Benchmark": benchmark_name,
    }

def format_model_name(model_name):
    """
    Format model filename for display in the sidebar card
    """
    clean_name = os.path.splitext(model_name)[0]
    clean_name = clean_name.replace("_model", "")
    clean_name = clean_name.replace("_pipeline", "")
    return clean_name

def render_sidebar(model_name):
    """
    Render sidebar and return prediction settings
    """
    with st.sidebar:
        st.markdown("#### Model")
        model_options = get_available_model_options()
        if "selected_model_file" not in st.session_state:
            st.session_state.selected_model_file = get_best_model_file()
        if "pending_model_file" not in st.session_state:
            st.session_state.pending_model_file = st.session_state.selected_model_file

        option_files = [model_file for model_file, _ in model_options]
        option_labels = {model_file: label for model_file, label in model_options}
        default_index = option_files.index(st.session_state.pending_model_file) if st.session_state.pending_model_file in option_files else 0

        pending_model_file = st.selectbox(
            "Model selector",
            options=option_files,
            index=default_index,
            format_func=lambda model_file: option_labels.get(model_file, model_file),
        )
        st.session_state.pending_model_file = pending_model_file

        if st.button("Load Model"):
            st.session_state.selected_model_file = st.session_state.pending_model_file

        active_model_file = st.session_state.selected_model_file

        st.markdown(
            f'<div class="model-card"><div class="model-card-name">{format_model_name(active_model_file)}</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("#### Prediction Settings")
        threshold = st.slider(
            "Real-news threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.05,
        )
        st.caption("Scores below threshold -> Fake News.")
        min_chars = st.number_input(
            "Min chars",
            min_value=10,
            max_value=500,
            value=30,
            step=10,
        )
        max_chars = st.number_input(
            "Max chars",
            min_value=1000,
            max_value=100000,
            value=20000,
            step=1000,
        )

        st.markdown("---")
        st.markdown("#### Live URL Settings")
        extraction_strategy = st.selectbox(
            "Extraction strategy",
            ["Trafilatura (precise)", "BeautifulSoup (fallback)"],
        )

        st.markdown("---")
        st.markdown("#### Metadata")
        metadata = load_sidebar_metadata(active_model_file)
        metadata_df = pd.DataFrame(metadata.items(), columns=["Item", "Value"])
        st.dataframe(metadata_df, hide_index=True, use_container_width=True)

        return active_model_file, float(threshold), int(min_chars), int(max_chars), extraction_strategy

def render_predict_tab(real_news_1, fake_news, selected_model_file, threshold, min_chars, max_chars):
    """
    Render the prediction tab
    """
    if "single_input" not in st.session_state:
        st.session_state.single_input = ""

    def set_sample(text):
        st.session_state.single_input = text

    col_a, col_b, col_c = st.columns([1, 1, 1])
    if col_a.button("Load Real Example"):
        set_sample(real_news_1)
    if col_b.button("Load Fake Example"):
        set_sample(fake_news)
    if col_c.button("Clear"):
        st.session_state.single_input = ""

    text = st.text_area(
        "Article Text",
        key="single_input",
        height=240,
        placeholder="Paste a full article or long excerpt for better reliability...",
    )

    if st.button("Analyze Article", type="primary"):
        ok, message = validate_text_input(text, min_chars, max_chars)
        if not ok:
            st.warning(message)
            return

        with st.spinner("Running prediction..."):
            try:
                result, confidence, probabilities, model_name, cleaned_text = predict_news(text, threshold, selected_model_file)
                render_prediction_card(result, confidence, probabilities, model_name)
                render_preprocessing_expander(cleaned_text)
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

def render_benchmark_tab():
    """
    Render the benchmark tab using stored notebook artifacts
    """
    st.caption("Loads benchmark results from the saved notebook artifact files.")

    if st.button("Load Stored Benchmark", type="primary"):
        try:
            results, result_file = load_benchmark_results()
        except Exception as e:
            st.error(f"Benchmark failed: {str(e)}")
            return

        st.caption(f"Loaded from: {result_file}")

        rows = []
        for model_name, model_data in results.items():
            if isinstance(model_data, dict):
                rows.append(
                    {
                        "Model": model_name,
                        "Accuracy": float(model_data.get("accuracy", 0)),
                        "Precision": float(model_data.get("precision", 0)),
                        "Recall": float(model_data.get("recall", 0)),
                        "F1": float(model_data.get("f1", 0)),
                        "CV Mean": float(model_data.get("cv_mean", 0)),
                    }
                )

        if not rows:
            st.warning("No benchmark rows were found in the saved results file.")
            return

        benchmark_df = pd.DataFrame(rows).sort_values("F1", ascending=False)
        st.dataframe(benchmark_df, hide_index=True, use_container_width=True)

        best_row = benchmark_df.iloc[0]
        cols = st.columns(4)
        cols[0].metric("Best Model", best_row["Model"])
        cols[1].metric("Accuracy", f"{best_row['Accuracy']:.4f}")
        cols[2].metric("Recall", f"{best_row['Recall']:.4f}")
        cols[3].metric("F1", f"{best_row['F1']:.4f}")

        chart_df = benchmark_df.melt(
            id_vars="Model",
            value_vars=["Accuracy", "Precision", "Recall", "F1"],
            var_name="Metric",
            value_name="Score",
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=chart_df, x="Metric", y="Score", hue="Model", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_title("Stored Benchmark Comparison")
        ax.set_ylabel("Score")
        st.pyplot(fig)

def extract_text_from_url(url, extraction_strategy):
    """
    Extract article text from a live URL
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        )
    }

    use_trafilatura = extraction_strategy.startswith("Trafilatura")
    if use_trafilatura and trafilatura is not None:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(
                downloaded,
                url=url,
                include_comments=False,
                include_tables=False,
                include_links=False,
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

    text = "\n".join(p for p in paragraphs if p)
    return text.strip()

def render_live_url_tab(selected_model_file, threshold, min_chars, max_chars, extraction_strategy):
    """
    Render the live URL tab
    """
    st.caption("Live URL mode performs model inference only. It does not verify ground-truth claims.")

    sample_url_map = {
        "Malay Mail (sample 1)": "https://www.malaymail.com/news/singapore/2025/09/10/yes-i-am-muslimchineseindian-but-im-also-singaporean-lee-hsien-loong-says-national-identity-matters-but-may-not-be-most-important/190617",
        "Malay Mail (sample 2)": "https://www.malaymail.com/news/world/2025/09/11/nepal-gen-z-backs-ex-chief-justice-sushila-karki-to-lead-after-pm-ousted-in-deadly-protests/190762",
        "Malay Mail (sample 3)": "https://www.malaymail.com/news/world/2025/09/11/not-a-friendly-place-what-life-looks-like-for-thaksin-in-bangkoks-notorious-klong-prem-prison/190768",
        "CNN USA (sample 1)": "https://edition.cnn.com/world/live-news/iran-war-us-israel-trump-03-11-26",
        "CNA Singapore (sample 1)": "https://www.channelnewsasia.com/singapore/singapaw-air-pets-flight-private-plane-airline-5981701",
    }

    with st.expander("Use Sample News URLs", expanded=True):
        selected_label = st.selectbox(
            "Sample URL",
            options=list(sample_url_map.keys()),
            key="live_url_sample_select",
        )
        selected_url = sample_url_map[selected_label]

        sample_col_1, sample_col_2, _sample_spacer = st.columns([1, 1, 2.2])
        if sample_col_1.button("Load Sample URL"):
            st.session_state.live_url_input = selected_url
        with sample_col_2:
            render_open_link("Open selected source", selected_url)
        st.caption(
            "Use direct article URLs for prediction. Section/home pages are blocked because they extract mixed headlines."
        )

    if "live_url_input" not in st.session_state:
        st.session_state.live_url_input = ""

    url = st.text_input(
        "News URL",
        key="live_url_input",
        placeholder="https://example.com/news/article",
    )

    action_col_1, action_col_2, _action_spacer = st.columns([1, 1, 2.2])
    with action_col_1:
        analyze_clicked = st.button("Fetch and Analyze URL", type="primary")
    with action_col_2:
        if url.strip().startswith(("http://", "https://")):
            render_open_link("Open current URL", url)
        else:
            render_open_link("Open current URL", "#", disabled=True)

    if analyze_clicked:
        if not url.strip().startswith(("http://", "https://")):
            st.warning("Please enter a valid http/https URL.")
            return

        with st.spinner("Fetching and extracting article text..."):
            try:
                extracted_text = extract_text_from_url(url, extraction_strategy)
            except Exception as e:
                st.error(f"URL fetch/extraction failed: {str(e)}")
                return

        ok, message = validate_text_input(extracted_text, min_chars, max_chars)
        if not ok:
            st.warning(message)
            st.text_area("Extracted Preview", value=extracted_text[:3000], height=220)
            return

        st.success("Article text extracted successfully.")
        st.text_area("Extracted Preview", value=extracted_text[:3000], height=220)

        try:
            result, confidence, probabilities, model_name, cleaned_text = predict_news(extracted_text, threshold, selected_model_file)
            render_prediction_card(result, confidence, probabilities, model_name)
            render_preprocessing_expander(cleaned_text)
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# Main application
def main():
    """
    Main function
    
    This function:
    1. Sets up the application interface
    2. Loads the model and vectorizer
    3. Creates the user interface layout
    4. Handle user interactions
    5. Displays prediction results
    
    The application used stramlit for the front-end
    """

    # Display the main Header 
    st.markdown('<div class="hero-title">News Authenticity Classifier</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Standalone fake-news project with robust validation workflows and artifact checks.</div>',
        unsafe_allow_html=True,
    )

    # Initialize session state for text input if it doesn;t exist 
    # Session state preserves values across reruns
    if "single_input" not in st.session_state:
        st.session_state.single_input = ""
    if "live_url_input" not in st.session_state:
        st.session_state.live_url_input = ""

    # Sample texts for demonstation
    
    # Real news example
    real_news_1 = """(Sourced From MalayMail 09/10/2025) From today onwards, Malaysia will no longer punish those who try to commit suicide with jail or fine, as it is no longer a crime in the country. Instead, the Malaysian government has improved its mental health law, by empowering more officers to rescue those who attempt suicide and quickly get medical help for them. In a joint statement by the Prime Minister’s Department’s Legal Affairs Division (BHEUU) and the Health Ministry (MOH), the government said it had today enforced three new laws in conjunction with World Suicide Prevention Day 2025. These three laws were passed in Parliament in 2023, but did not take effect until today. “It is the Madani government’s hope that the reforms of these laws will be a huge shift in efforts to prevent attempted suicides in Malaysia, by encouraging those whose mental health are affected, to step forward to get help; to eradicate stigma towards attempted suicides and reduce the rate of deaths due to suicides,” BHEUU and MOH said in the statement."""

    # Fake news example
    fake_news = """BREAKING: Government Secretly Installing Mind Control Devices in COVID Vaccines. In a shocking revelation, anonymous sources within the Pentagon have confirmed that the government is using COVID-19 vaccines to implant microscopic tracking and mind control devices in citizens. These nano-chips, developed by Bill Gates and funded by global elites, can monitor your thoughts and movements 24/7. The devices are activated by 5G towers that have been strategically placed across the country. People who received the vaccine report strange dreams and sudden urges to obey government mandates. One victim reported, After my second dose, I suddenly wanted to eat more vegetables and exercise daily - something I never did before! Doctors who have spoken out against this conspiracy have mysteriously disappeared. The mainstream media is covering up this scandal despite overwhelming evidence. Protect yourself by refusing vaccination and shielding your home with aluminum foil to block 5G signals.Share this urgent news before it gets censored! The truth must be revealed!"""

    # Load model information for sidebar
    default_model_file = get_best_model_file()
    sidebar_model_name = default_model_file if default_model_file else "No model loaded"
    selected_model_file, threshold, min_chars, max_chars, extraction_strategy = render_sidebar(sidebar_model_name)

    # Create three tabs similar to the newer frontend
    tab_predict, tab_benchmark, tab_live = st.tabs(
        ["📝 Predict", "📊 Benchmark", "🔗 Live URL"]
    )

    with tab_predict:
        render_predict_tab(real_news_1, fake_news, selected_model_file, threshold, min_chars, max_chars)

    with tab_benchmark:
        render_benchmark_tab()

    with tab_live:
        render_live_url_tab(selected_model_file, threshold, min_chars, max_chars, extraction_strategy)

# Entry point of the app
if __name__ == "__main__":
    main()

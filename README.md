# Fake News Classification

This project implements a professor-aligned fake news classification workflow using classical machine learning for binary text classification. The main deliverables are a fully structured training notebook, a synchronized Streamlit deployment app, and a clean repository layout where datasets and trained artifacts live in dedicated folders.

## Project Highlights

- End-to-end notebook organized around a clear 11-stage pipeline.
- Enhanced exploratory data analysis with consistent visual styling.
- Explicit preprocessing stages: normalization, cleaning, tokenization, stopword removal, and POS-aware lemmatization.
- TF-IDF vector representation for classical machine learning.
- Baseline model comparison across Logistic Regression, Naive Bayes, Linear SVM, and Random Forest.
- Hyperparameter tuning with `GridSearchCV` and `RandomizedSearchCV`.
- Deployment app synchronized with the notebook pipeline.
- All trained `.pkl` artifacts exported to the `Models` folder only.

## Primary Files

- `News_Classification.ipynb`
  The main training and evaluation notebook.
- `AppNewsClassification.py`
  The primary deployment app aligned with the notebook pipeline.
- `App.py`
  An older app variant kept in the repository for reference.
- `requirements.txt`
  Python dependencies captured from the environment.
- `sample.txt`
  Example article text for quick app testing.

## Professor-Aligned Pipeline

The notebook is organized to follow the required workflow explicitly:

| Stage | Name | Purpose |
| --- | --- | --- |
| 01 | Problem Definition and Dataset Acquisition | Define the task and load the fake/real news datasets |
| 02 | Data Understanding | Inspect class balance, text characteristics, and dataset structure |
| 03 | Data Cleaning | Remove duplicates, handle missing values, and prepare raw fields |
| 04 | Text Preprocessing | Normalize and transform text into cleaned tokens and lemmas |
| 05 | Data Split | Create a stratified train/test split |
| 06 | Word Vector Representation | Convert cleaned text into TF-IDF vectors |
| 07 | ML Algorithm Building | Train and benchmark baseline classical ML models |
| 08 | Hyperparameter Tuning | Tune shortlisted models using cross-validation |
| 09 | Tuned Model Selection | Select the strongest final model |
| 10 | Evaluation | Report confusion matrix and classification metrics |
| 11 | Sample Prediction | Run inference on new example text |

## Preprocessing Flow

The preprocessing logic used in the notebook and the main app is synchronized:

1. Text normalization
2. Text cleaning
3. Tokenization
4. Stopword removal
5. POS-aware lemmatization
6. Final cleaned text generation
7. TF-IDF vectorization
8. Model prediction

## Models Evaluated

The notebook benchmarks the following classifiers:

- Logistic Regression
- Naive Bayes
- Linear SVM
- Random Forest

Validation uses `StratifiedKFold`, with additional repeated stratified cross-validation diagnostics for the strongest baseline candidates.

## Final Notebook Results

The latest saved notebook artifacts identify **Linear SVM** as the final selected model.

| Model | Stage | Search Method | Accuracy | Precision | Recall | F1 Score | CV Mean |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | Tuned | GridSearchCV | 0.9944 | 0.9923 | 0.9974 | 0.9948 | 0.9945 |
| Naive Bayes | Tuned | GridSearchCV | 0.9606 | 0.9659 | 0.9613 | 0.9636 | 0.9646 |
| Linear SVM | Tuned | RandomizedSearchCV | 0.9967 | 0.9948 | 0.9991 | 0.9969 | 0.9967 |
| Random Forest | Baseline | Baseline | 0.9941 | 0.9918 | 0.9974 | 0.9946 | 0.9943 |

Best saved model metadata:

- Best model: `Linear SVM`
- Saved slug: `linear_svm`
- TF-IDF configuration:
  `max_features=10000`, `ngram_range=(1, 2)`, `min_df=5`, `max_df=0.85`, `sublinear_tf=True`

## Repository Structure

```text
Fake-News-Classification/
├── App.py
├── AppNewsClassification.py
├── Dataset/
├── Models/
├── News_Classification.ipynb
├── README.md
├── requirements.txt
└── sample.txt
```

## Important Folder Note

The GitHub repository keeps `Dataset` and `Models` intentionally empty except for placeholder files.

This is intentional:

- `Dataset/` is where you should place the input CSV files such as `fake.csv` and `true.csv`.
- `Models/` is where the notebook exports trained artifacts such as:
  `best_model_pipeline.pkl`, `best_model_estimator.pkl`, `best_model_metadata.pkl`, `benchmark_results.pkl`, and `tfidf_vectorizer.pkl`.

The deployment app reads artifacts from `Models` only.

## How to Reproduce the Workflow

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

### 2. Prepare the dataset

Place the dataset files inside `Dataset/`. The notebook expects the fake and real news CSV files there.

### 3. Run the notebook

Open and run:

```text
News_Classification.ipynb
```

The notebook will:

- perform EDA
- preprocess the text
- train and compare models
- tune shortlisted models
- evaluate the best model
- export trained `.pkl` files into `Models/`

### 4. Run the deployment app

Use the professor-aligned app:

```powershell
streamlit run AppNewsClassification.py
```

## App Features

`AppNewsClassification.py` includes:

- text-based fake/real news prediction
- URL-based article extraction
- adjustable real-news threshold
- synchronized preprocessing trace display
- benchmark view using saved notebook results
- pipeline overview tab
- artifact loading from `Models` only

## Saved Artifacts

After a successful notebook run, the `Models` folder should contain artifacts such as:

- `benchmark_results.pkl`
- `best_model_estimator.pkl`
- `best_model_metadata.pkl`
- `best_model_pipeline.pkl`
- `tfidf_vectorizer.pkl`
- model-specific pipeline and estimator files

## Submission Notes

For presentation or submission, the recommended application entry point is:

```text
AppNewsClassification.py
```

This file is the version aligned with the notebook’s professor-required pipeline.

## License

This repository currently includes the existing `LICENSE` file from the GitHub project root.

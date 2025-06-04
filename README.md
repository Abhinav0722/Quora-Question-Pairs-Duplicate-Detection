# Quora Question Pairs Duplicate Detection

A complete end-to-end machine learning pipeline that detects whether two free-text questions are duplicates. This repository includes data preprocessing, feature engineering, model training, threshold tuning, and model comparison—all in a single `main.py` script.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Technology Stack](#technology-stack)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Directory Structure](#directory-structure)  
7. [Configuration & Environment Variables](#configuration--environment-variables)  
8. [Model Training & Evaluation](#model-training--evaluation)  
9. [Contributing](#contributing)  
10. [License](#license)  

---

## Project Overview

Large Q&A platforms often face the problem of redundant questions—users asking the same thing in slightly different words. This leads to fragmented answers and extra moderation work. This project builds a pipeline that:

1. Samples and preprocesses raw question-pair data.  
2. Computes clean text, semantic embeddings, and additional features.  
3. Trains multiple classifiers (LightGBM, XGBoost, CatBoost, RandomForest, LogisticRegression).  
4. Sweeps decision thresholds to maximize F1-score.  
5. Compares model performance and outputs a summary table.

---

## Features

- **Data Sampling & Cleaning**  
  - Randomly sample 30,000 rows from the Quora QP dataset.  
  - Expand English contractions, normalize Unicode, strip HTML, replace symbols (%, $, ₹) with words.  

- **Feature Engineering**  
  - Compute sentence embeddings using `all-MiniLM-L6-v2`.  
  - Calculate cosine similarity between question-pair embeddings.  
  - Reduce absolute embedding-difference vectors via Incremental PCA (50 components).  
  - Compute token-overlap ratio (common tokens / min token count).  

- **Modeling & Threshold Tuning**  
  - Train LightGBM with class-weighting to handle duplicates vs. non-duplicates imbalance.  
  - Sweep thresholds from 0.50 to 0.99 to pick the one that maximizes F1.  
  - Train and evaluate RandomForest, XGBoost, LogisticRegression, and CatBoost similarly.  
  - Output confusion matrices, classification reports, and a model comparison table (sorted by F1).  

---

## Technology Stack

- **Core Libraries**:  
  - Python 3.9+  
  - pandas  
  - NumPy  
  - scikit-learn  
  - matplotlib  
  - BeautifulSoup (for HTML stripping)  
  - unicodedata (for Unicode normalization)

- **NLP & Embeddings**:  
  - sentence-transformers (all-MiniLM-L6-v2)

- **Machine Learning Frameworks**:  
  - LightGBM  
  - XGBoost  
  - CatBoost  
  - scikit-learn (RandomForest, LogisticRegression)

- **Visualization**:  
  - matplotlib (for confusion matrix plots)
  - seaborn (for pairplots)

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/quora-duplicate-detection.git
   cd quora-duplicate-detection

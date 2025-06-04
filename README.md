# 🤖 Quora Duplicate Question Detection

An end-to-end ML pipeline that identifies whether two free-text questions are semantic duplicates. By leveraging sentence-transformers embeddings and gradient-boosted models, this project streamlines Q&A platforms, reducing redundancy and improving user experience.

---

## 🎯 Project Summary

This repository processes Quora Question Pairs to detect duplicates. It samples 30K pairs, cleans and normalizes text (HTML stripping, contraction expansion, Unicode normalization), generates embeddings (all-MiniLM-L6-v2), and engineers features (cosine similarity, PCA-reduced embedding differences, token overlap). Multiple classifiers (LightGBM, XGBoost, CatBoost, RandomForest, LogisticRegression) are trained with class-weighting and threshold sweeps to maximize F₁-score. Final outputs include confusion matrices, classification reports, and a comparative summary.

---

## 🧠 Key Highlights

- 🧹 **Advanced Text Preprocessing**: Contraction expansion, Unicode normalization, HTML stripping.  
- 🤖 **Semantic Embeddings**: all-MiniLM-L6-v2 for 384-dim sentence vectors.  
- 📐 **Feature Engineering**:  
  - Cosine similarity between Q1/Q2 embeddings  
  - Incremental PCA (50 components) on absolute embedding differences  
  - Token-overlap ratio (common tokens ÷ min token count)  
- 🔍 **Modeling & Threshold Tuning**:  
  - LightGBM (class-weighted) → Best F₁ at optimized threshold  
  - XGBoost, CatBoost, RandomForest, LogisticRegression for comparison  
- 📊 **Evaluation & Visualization**: Confusion matrices and classification reports for each model.  

---

## 🚀 Technologies & Tools

| Category               | Tools / Libraries                              |
|------------------------|------------------------------------------------|
| Language               | Python 3.9+                                    |
| Data Handling          | pandas, NumPy                                  |
| Text Cleaning          | BeautifulSoup, unicodedata                     |
| Embeddings             | sentence-transformers (all-MiniLM-L6-v2)       |
| ML Models              | LightGBM, XGBoost, CatBoost, scikit-learn      |
| Dimensionality Reduction | scikit-learn (IncrementalPCA)                |
| Visualization          | matplotlib                                     |

---

## 📌 Project Workflow

### 1. Data Sampling & Cleaning
- Load `train.csv` with ~400K Quora QP entries.  
- Randomly sample 30,000 rows for faster iteration.  
- Expand contractions (e.g., “can’t” → “can not”), normalize Unicode (e.g., “résumé” → “resume”), strip any HTML tags.

### 2. Embedding Generation
- Collect unique cleaned questions (~60K).  
- Use `sentence-transformers` to compute 384-dim embeddings for each unique question.  

### 3. Feature Engineering
- **Cosine Similarity**: Compute similarity between Q1 and Q2 embeddings (single scalar).  
- **Embedding Difference + PCA**:  
  - Compute absolute difference vectors: |E₁ – E₂| ∈ ℝ³⁸⁴.  
  - Run IncrementalPCA (n_components=50) in batches of 5,000.  
- **Token Overlap Ratio**: (|set(tokens₁) ∩ set(tokens₂)|) ÷ min(|tokens₁|, |tokens₂|).

### 4. Model Training & Threshold Optimization
- Split features into train (80%) / test (20%) with stratification on `is_duplicate`.  
- Compute `base_weight = (#non_duplicates)/(#duplicates)` and apply `class_weight={0:1.0, 1:base_weight×3.0}` for LightGBM.  
- Train **LightGBMClassifier** (n_estimators=200, learning_rate=0.05, max_depth=6).  
- Predict probabilities on test set, sweep thresholds from 0.50 to 0.99 (40 steps) to maximize F₁.  
- Repeat above for **RandomForestClassifier**, **XGBClassifier**, **LogisticRegression**, and **CatBoostClassifier** (200 iterations, depth=6, learning_rate=0.05, same class weights).  
- For each model:  
  - Record best threshold, TP, FP, TN, FN, precision, recall, F₁.  
  - Plot confusion matrix at best threshold.  

### 5. Model Comparison
- Compile summary table (`model`, `threshold`, `TP`, `FP`, `TN`, `FN`, `precision`, `recall`, `f1`) sorted by descending F₁.  
- Identify top-performing model for deployment or further tuning.

---

## 📈 Performance Comparison

| Model                | Best Threshold | Precision | Recall | F₁-score |
|----------------------|----------------|-----------|--------|----------|
| LightGBM             | 0.72           | 0.85      | 0.74   | 0.79     |
| CatBoost             | 0.68           | 0.86      | 0.72   | 0.78     |
| XGBoost              | 0.75           | 0.84      | 0.70   | 0.76     |
| RandomForest         | 0.50           | 0.80      | 0.68   | 0.73     |
| LogisticRegression   | 0.65           | 0.77      | 0.64   | 0.70     |

*Note: Metrics are illustrative; actual results vary based on random seed and sampled data.*

---

## 🌍 Real-World Applications

- **Q&A Platforms**: Automatically flagging duplicate questions reduces content fragmentation and moderation effort.  
- **Customer Support**: Detect repeated tickets or FAQs to surface existing solutions quickly.  
- **Knowledge Base Management**: Consolidate similar articles or posts to maintain a clean, searchable repository.  
- **Community Forums**: Improve user experience by merging near-identical threads.

---

## 🔧 Future Enhancements

- 🌐 **Multilingual Support**: Integrate multilingual embeddings (e.g., XLM-RoBERTa) to handle non-English question pairs.  
- 🔄 **Active Learning**: Implement a human-in-the-loop for ambiguous pairs (probability ~0.45–0.55) to continuously improve the model.  
- 📊 **Explainability**: Add SHAP or LIME to highlight which features (e.g., token overlap vs. embedding differences) contributed most to each prediction.  
- 🛠 **End-to-End Deployment**: Build a lightweight Flask or FastAPI service for real-time inference, containerize with Docker, and set up CI/CD.  
- 🧪 **Data Augmentation**: Use paraphrasing techniques (back-translation, T5-based paraphraser) to expand training data and improve generalization.

---

## 🤝 Contributing

1. Fork this repository.  
2. Create a feature branch:  
   ```bash
   git checkout -b feature/YourFeatureName

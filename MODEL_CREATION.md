# Model Creation Details (Assignment Requirement)

This document explains the model creation process used in this project and maps directly to the assignment tasks.

## 1) Data Gathering and Cleaning
- Sources:
  - `data/drugsComTrain_raw.csv`
  - `data/drugsComTest_raw.csv`
- Steps:
  - Load CSV datasets with pandas
  - Remove rows missing `condition`, `review`, or `rating`
  - Remove duplicate rows
  - Keep rating in valid range `[1, 10]`
  - Convert `date` to datetime (`errors="coerce"`)
- Outputs:
  - cleaned train/test datasets
  - descriptive statistics printed by training script
  - `artifacts/data_quality_summary.csv`
  - rating outlier counts using the IQR rule
  - useful count outlier counts using the IQR rule

## 2) Data Preprocessing
- Condition filtering:
  - Keep only target conditions:
    - Depression
    - High Blood Pressure
    - Type 2 Diabetes
  - Map dataset labels (e.g., `Diabetes, Type 2`) to standardized target labels
- Text preprocessing:
  - lowercasing
  - punctuation and number removal
  - whitespace normalization
  - tokenization + stopword removal using NLTK
- Feature extraction:
  - TF-IDF vectorization
- Feature tuning:
  - compares 3,000 unigram features
  - compares 5,000 unigram features
  - compares 5,000 unigram + bigram features
  - compares 8,000 unigram + bigram features
  - selects the best validation-accuracy configuration for final training
  - saves results to `artifacts/feature_tuning_results.csv`

## 3) Model Determination
Candidate models:
- Logistic Regression
- Multinomial Naive Bayes

Reasoning:
- Both models are strong baselines for sparse TF-IDF text features.
- Logistic Regression usually gives strong multi-class decision boundaries.
- Multinomial Naive Bayes is fast and robust for word-frequency style features.

Model selection is based on:
- Validation accuracy
- Cross-validation accuracy
- Test accuracy

## 4) Model Training
- Split: train/validation using stratified split.
- Fit both candidate models.
- Compute:
  - Validation accuracy
  - Cross-validation accuracy (5-fold)
  - Test accuracy
- Classification report
- Confusion matrix
- Baseline performance matrix saved to `artifacts/model_results.csv`

## 5) Model Tuning
- `GridSearchCV` with 5-fold CV:
  - Logistic Regression: `C`, `solver`, `penalty`
  - Multinomial Naive Bayes: `alpha`
- Best tuned model chosen by test accuracy.
- Tuned model performance saved to `artifacts/tuned_model_results.csv`.
- Tuned classification reports saved under `artifacts/`.

## 6) Saved Artifacts
Saved to `models/`:
- `model.pkl` (best tuned model)
- `vectorizer.pkl`
- `label_encoder.pkl`

Saved to `artifacts/plots/`:
- condition distribution
- rating distribution
- rating boxplots
- useful count distribution
- monthly date trends for review volume and average rating
- top drugs by average useful count
- feature tuning comparison
- word cloud
- confusion matrices
- model comparison

## 7) Reproducibility Command
Run:
```cmd
cd /d C:\Users\reabe\Desktop\AIDrugReview
.\.venv\Scripts\python training\main.py
```

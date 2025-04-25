# Credit Risk Prediction

## Overview

This project predicts credit risk using the German Credit dataset with 1000 records, 20 features, and a binary target: credit_risk (1 for good, 2 for bad). Implemented in Google Colab using a Random Forest Classifier, it achieves 80.5 percent accuracy and provides dynamic recommendations for credit evaluation.

## Why Random Forest

The Random Forest Classifier was selected for its strengths. It handles imbalanced data (70 percent good, 30 percent bad) effectively. It provides feature importance to identify key risk factors. It achieves approximately 80 percent accuracy, as seen with 80.5 percent in this project. It supports dynamic recommendations based on model outputs. Alternatives like XGBoost require more tuning and are less interpretable, SVM is sensitive to imbalance, and logistic regression is unsuitable for non-linear patterns.

## Methodology

### Data Exploration

The dataset has 1000 instances, 20 features (7 numerical, 13 categorical), and 1 target. Class distribution shows 700 good (Class 1) and 300 bad (Class 2) cases, with a ratio of 0.43. A count plot highlighted class imbalance. A heatmap showed moderate correlations, such as duration and credit_amount at approximately 0.62. Key features include checking_account, credit_amount, and duration.

### Data Preprocessing

Categorical features like checking_account were encoded to numerical values using LabelEncoder. Feature engineering created credit_amount_to_duration (loan amount to duration ratio) and age_group (binned age: 0-30, 30-50, 50+). Numerical features were scaled with StandardScaler. The dataset had no missing values, and outliers were retained due to Random Forest’s robustness.

### Model Development

The model is a Random Forest Classifier. Data was split 80 percent training, 20 percent testing (random_state=42). GridSearchCV with 5-fold cross-validation tuned n_estimators (100, 200), max_depth (10, 20, None), min_samples_split (2, 5), and used F1-score for scoring. Metrics include accuracy, precision, recall, and F1-score, with focus on Class 2 recall.

### Model Interpretation

Feature importance was visualized with a bar plot. Recommendations were generated based on top features, Class 2 recall, and imbalance ratio.

## Results

### Data Exploration

Class distribution is 70 percent good, 30 percent bad (ratio: 0.43). Visualizations supported non-linear modeling and feature engineering.

### Model Performance

On the test set (200 samples), the model achieved: Accuracy 0.8050, Precision 0.8187, Recall (Class 1) 0.9291, F1-Score 0.8704. The classification report shows: Class 1 (Good) with Precision 0.82, Recall 0.93, F1-Score 0.87 (141 samples); Class 2 (Bad) with Precision 0.75, Recall 0.51, F1-Score 0.61 (59 samples); Macro Avg with Precision 0.78, Recall 0.72, F1-Score 0.74; Weighted Avg with Precision 0.80, Recall 0.81, F1-Score 0.79. The model excels for good credit risks but has low Class 2 recall (0.51), risking missed bad credit applicants. Accuracy (80.5 percent) meets benchmarks.

### Feature Importance

Top features are checking_account (financial stability), credit_amount_to_duration (loan burden), and credit_amount (loan size).

### Recommendations

1. Prioritize checking_account, credit_amount_to_duration, credit_amount for risk assessment.
2. Monitor class distribution (ratio: 0.43).
3. Improve Class 2 recall (0.51) with XGBoost or threshold adjustments.
4. Add features like income or credit scores.

## Conclusions

The Random Forest model achieves 80.5 percent accuracy but has a low Class 2 recall (0.51), risking missed bad credit risks. Top features guide efficient evaluation, and dynamic recommendations address recall and data limitations. Future work should boost Class 2 recall.

## Future Work

Improve Class 2 recall with SMOTE or XGBoost. Add features like income or credit scores. Test Gradient Boosting or Neural Networks. Enhance interpretability with SHAP or LIME. Optimize thresholds for Class 2 recall.

## Setup and Usage

### Prerequisites

Google Colab and libraries: pandas, numpy, scikit-learn, matplotlib, seaborn.

### Instructions

1. Open a new Colab notebook.
2. Install libraries:
   ```bash
   !pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Copy and run credit_risk_prediction.py from the repository.
4. View outputs: dataset info, count plot, heatmap, model metrics, classification report, feature importance plot, recommendations.

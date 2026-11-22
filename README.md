# AI-Powered Customer Support Intelligence System

An experimental machine learning project focused on analyzing customer support data and predicting Customer Satisfaction (CSAT) scores using Python, Scikit-learn, Natural Language Processing (NLP), and feature engineering.

The project explores operational ticket information, customer remarks, numerical features, and missing-value patterns to understand their relationship with customer satisfaction.

## Project Overview

Customer support teams handle large volumes of customer requests across different channels and categories. Understanding customer satisfaction can help identify service issues and improve support operations.

This project investigates how machine learning can classify customer satisfaction scores from historical support records.

**Current implementation:** Data cleaning, exploratory data analysis, feature engineering, Logistic Regression training, model evaluation, and performance comparison.

**Planned development:** Advanced ML models, a React frontend, FastAPI backend, knowledge-base RAG, and AI-generated response suggestions.

## Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Data processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| NLP | TF-IDF |
| Feature encoding | OneHotEncoder |
| Feature scaling | StandardScaler |
| Model persistence | Joblib |
| Data visualization and analysis | Pandas |
| Version control | Git, GitHub |

## Dataset Information

The project currently uses a customer support dataset containing:

| Dataset property | Value |
|---|---:|
| Total records | 85,907 |
| Original columns | 20 |
| Columns after cleaning | 21 |
| Training records | 68,725 |
| Testing records | 17,182 |
| Training split | 80% |
| Testing split | 20% |
| Random state | 42 |

### Target Variable

The target variable is `CSAT Score`, with five possible classes:

- 1 – Lowest satisfaction score
- 2 – Low satisfaction score
- 3 – Moderate satisfaction score
- 4 – High satisfaction score
- 5 – Highest satisfaction score

### CSAT Distribution

| CSAT Score | Records | Distribution |
|---|---:|---:|
| 1 | 11,230 | 13.07% |
| 2 | 1,283 | 1.49% |
| 3 | 2,558 | 2.98% |
| 4 | 11,219 | 13.06% |
| 5 | 59,617 | 69.40% |
| **Total** | **85,907** | **100%** |

The dataset is imbalanced, with CSAT score 5 representing approximately 69.40% of all records. Therefore, Macro F1 and balanced accuracy are important evaluation metrics alongside ordinary accuracy.

## Data Preprocessing

The preprocessing pipeline includes:

1. Loading the original CSV dataset.
2. Cleaning customer remarks.
3. Removing HTML tags and unwanted text patterns.
4. Normalizing whitespace.
5. Handling missing categorical values.
6. Converting numerical features into appropriate data types.
7. Separating training and testing data using stratified sampling.

### Text Cleaning

The `Customer Remarks` column is cleaned and stored in `customer_text`.

The cleaning process uses Python, Pandas, and regular expressions.

### Data Quality Findings

| Column | Missing records | Missing percentage |
|---|---:|---:|
| Customer Remarks | 57,165 | Approximately 66.55% |
| Item_price | 68,701 | 79.97% |
| connected_handling_time | 85,665 | 99.72% |
| Order_id | Approximately 21.22% | 21.22% |
| order_date_time | Approximately 79.96% | 79.96% |
| Customer_City | Approximately 80.12% | 80.12% |

The handling-time feature was excluded from the current experiments because almost all its values were missing.

## Feature Engineering and Analysis

The project investigates the following operational features:

- Category
- Sub-category
- Channel name
- Tenure Bucket
- Agent Shift

Numerical features investigated:

- Item_price
- connected_handling_time

An additional feature was created:

`item_price_missing`

This binary feature indicates whether an item's price is missing.

- 0: Item price is available.
- 1: Item price is missing.

### Feature Association Analysis

Normalized Mutual Information (NMI) was used to investigate the association between operational features and CSAT.

| Feature | NMI |
|---|---:|
| Sub-category | 0.0099 |
| Category | 0.0041 |
| Channel name | 0.0012 |
| Tenure Bucket | 0.0012 |
| Agent Shift | 0.0011 |

The results indicate weak individual associations between the investigated operational features and CSAT.

### Numerical Feature Analysis

| Feature | Available records | Missing percentage | Correlation with CSAT |
|---|---:|---:|---:|
| Item_price | 17,206 | 79.97% | -0.1233 |
| connected_handling_time | 242 | 99.72% | 0.0475 |

Item price showed a weak negative linear correlation with CSAT. Handling time showed almost no linear correlation and was excluded from the current model.

Missing-value analysis also revealed that item-price availability varies substantially across support categories.

## Machine Learning Experiments

Three main Logistic Regression configurations were evaluated.

### Model 1: TF-IDF + Operational Features

Features:
- Customer remarks converted into TF-IDF vectors.
- Operational categorical features encoded using OneHotEncoder.
- Logistic Regression with balanced class weights.

TF-IDF configuration:
- Maximum features: 50,000
- N-gram range: (1, 2)
- Minimum document frequency: 2
- Sublinear TF: Enabled

**Test results:**

| Metric | Score |
|---|---:|
| Accuracy | 34.91% |
| Balanced Accuracy | 29.81% |
| Macro Precision | 0.3042 |
| Macro Recall | 0.2981 |
| Macro F1 | 0.2478 |
| Weighted F1 | 0.4341 |
| Training time | 13.59 seconds |
| Prediction time | 0.193 seconds |

Note: Customer remarks may contain post-interaction survey feedback. This creates a potential target-leakage risk if the intended prediction is made before customer feedback is received. This model is therefore an experimental comparison, not a validated pre-interaction prediction model.

### Model 2: Operational Features Only

Features:
- Category
- Sub-category
- Channel name
- Tenure Bucket
- Agent Shift

Model: Logistic Regression with balanced class weights.

**Test results:**

| Metric | Score |
|---|---:|
| Accuracy | 29.55% |
| Balanced Accuracy | 24.08% |
| Macro Precision | 0.2254 |
| Macro Recall | 0.2408 |
| Macro F1 | 0.1762 |
| Weighted F1 | 0.3660 |
| Prediction time | 0.069 seconds |

### Model 3: Operational Features + Item Price + Missing Indicator

This experiment adds:
- Item_price
- item_price_missing

Numerical features are imputed using the median and standardized using StandardScaler. Categorical features are imputed and encoded using OneHotEncoder.

**Final test results:**

| Metric | Score |
|---|---:|
| Accuracy | 33.58% |
| Balanced Accuracy | 25.77% |
| Macro Precision | 0.2369 |
| Macro Recall | 0.2577 |
| Macro F1 | **0.2015** |
| Weighted F1 | 0.4085 |
| Training time | 2.999 seconds |
| Prediction time | 0.041 seconds |
| Average prediction time | Approximately 2.38 microseconds per record |

This configuration achieved the best Macro F1 among the operational-only experiments conducted so far.

The improvement over the operational-only model was approximately 14.35% in relative Macro F1.

## Model Comparison

All values below are from the held-out test evaluation.

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Majority baseline | 69.40% | Not recorded | 0.1639 | Not recorded |
| TF-IDF + Operational | 34.91% | 29.81% | 0.2478 | 0.4341 |
| Operational only | 29.55% | 24.08% | 0.1762 | 0.3660 |
| Operational + Price + Missing Indicator | 33.58% | 25.77% | 0.2015 | 0.4085 |

The majority baseline predicts the most frequent class. Its high accuracy is a consequence of class imbalance and should not be interpreted as good performance across all five classes.

## Detailed Classification Report

### Best Operational Feature Configuration

The following classification report corresponds to the price-enhanced model.

| CSAT Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 1 | 0.21 | 0.36 | 0.27 | 2,246 |
| 2 | 0.02 | 0.29 | 0.04 | 256 |
| 3 | 0.04 | 0.15 | 0.07 | 512 |
| 4 | 0.14 | 0.12 | 0.13 | 2,244 |
| 5 | 0.77 | 0.38 | 0.51 | 11,924 |
| **Macro Average** | **0.24** | **0.26** | **0.20** | **17,182** |
| **Weighted Average** | **0.58** | **0.34** | **0.41** | **17,182** |

The model performs better on classes 1 and 5 than on classes 2 and 3. Minority-class prediction remains a major challenge.

### Confusion Matrix

Rows represent actual CSAT classes; columns represent predicted classes.

| Actual / Predicted | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| 1 | 801 | 617 | 214 | 180 | 434 |
| 2 | 62 | 73 | 39 | 27 | 55 |
| 3 | 113 | 150 | 75 | 44 | 130 |
| 4 | 473 | 508 | 248 | 266 | 749 |
| 5 | 2,283 | 2,486 | 1,219 | 1,382 | 4,554 |

## Performance Evaluation

The price-enhanced Logistic Regression model was trained and evaluated on the same 80/20 stratified split.

| Performance measurement | Result |
|---|---:|
| Training time | 2.999 seconds |
| Total prediction time | 0.040946 seconds |
| Test records | 17,182 |
| Average prediction time per record | 0.000002383 seconds |

These are local batch-processing measurements from the current environment. They do not represent complete application response time, deployment latency, or performance on other hardware.

## Project Structure

```text
AI-Powered-Customer-Support-Intelligence-System/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── clear_dataset.py
│   ├── analyze_features.py
│   ├── analyze_target_relationships.py
│   ├── analyze_numeric_features.py
│   ├── analyze_missingness.py
│   ├── prepare_dataset.py
│   ├── train_model.py
│   ├── train_operational_model.py
│   ├── evaluate_operational_model.py
│   ├── compare_price_features.py
│   └── evaluate_price_model.py
│
├── models/
│   ├── csat_logistic_regression.joblib
│   ├── csat_operational_model.joblib
│   └── csat_price_model.joblib
│
├── evaluation/
│   ├── metrics.json
│   ├── classification_report.txt
│   ├── operational_test_metrics.json
│   ├── operational_classification_report.txt
│   ├── price_model_test_metrics.json
│   └── price_model_classification_report.txt
│
├── backend/
├── frontend/
├── .gitignore
└── README.md
```

The raw and processed datasets are excluded from Git tracking to avoid publishing customer-related information.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/3161-harshit/AI-Powered-Customer-Support-Intelligence-System.git
```

### 2. Navigate into the project

```bash
cd AI-Powered-Customer-Support-Intelligence-System
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install pandas numpy scikit-learn joblib
```

### 5. Place the dataset

Place the authorized dataset at:

```text
data/raw/Customer_support_data.csv
```

### 6. Run preprocessing and experiments

```bash
python src/clear_dataset.py
python src/analyze_features.py
python src/prepare_dataset.py
python src/train_model.py
python src/train_operational_model.py
python src/evaluate_operational_model.py
python src/analyze_target_relationships.py
python src/analyze_numeric_features.py
python src/analyze_missingness.py
python src/compare_price_features.py
python src/evaluate_price_model.py
```

Run the scripts in the order appropriate to their input files.

## Current Limitations

1. **Class imbalance:** CSAT classes 2 and 3 have relatively few examples.
2. **Missing values:** Item price is missing in approximately 80% of records.
3. **Weak operational associations:** Individual operational features have low NMI values.
4. **Potential text leakage:** Customer remarks may contain feedback recorded after the outcome.
5. **Feature availability:** Item-price availability and other fields need to be verified against the intended prediction time.
6. **Model performance:** Current Macro F1 scores indicate that substantial improvement is still required.
7. **Deployment:** Production API, frontend, and real-time integration are not yet implemented.

## Future Development Roadmap

- [ ] Experiment with Random Forest and XGBoost.
- [ ] Investigate class imbalance and appropriate resampling strategies.
- [ ] Perform feature importance and error analysis.
- [ ] Validate features against prediction-time availability.
- [ ] Investigate data leakage and improve evaluation methodology.
- [ ] Build a FastAPI backend.
- [ ] Develop a React-based dashboard.
- [ ] Integrate a knowledge-base Retrieval-Augmented Generation (RAG) system.
- [ ] Explore AI-generated customer support response suggestions.
- [ ] Add monitoring and model performance tracking.

## Conclusion

This project explores an end-to-end experimental machine learning workflow for customer support analytics, from data cleaning and feature analysis to model training and evaluation.

The current experiments show that feature engineering, particularly the item-price missing-value indicator, can improve Macro F1 compared with the operational-only baseline.

However, the model still struggles with minority CSAT classes. Further experimentation, leakage checks, and feature validation are necessary before considering real-world deployment.

## Repository

GitHub: https://github.com/3161-harshit/AI-Powered-Customer-Support-Intelligence-System

---

**Project status:** Machine Learning experimentation and evaluation in progress.

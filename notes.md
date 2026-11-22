# AI-Powered Customer Support Intelligence System — Detailed Project Notes

Repository: https://github.com/3161-harshit/AI-Powered-Customer-Support-Intelligence-System

**Purpose:** A comprehensive research log for the dataset, preprocessing, exploratory analysis, machine-learning experiments, evaluation, limitations, and future application roadmap.

**Evidence policy:** Numerical results below are the values reported during the project work. They should be rerun if the source dataset, code, split, or dependencies change. Proposed application components are explicitly marked as future work.

## Table of contents
1. Project summary
2. Dataset facts
3. Data preparation
4. Feature analysis
5. Model experiments
6. Evaluation and performance
7. Repository and reproducibility
8. Risks and limitations
9. Future architecture
10. Detailed topic reference
11. Recommended next steps

## 1. Project summary

- The current implemented work is an exploratory CSAT prediction study using Python and Scikit-learn.
- The observed dataset contains 85,907 rows and 20 original columns.
- Cleaning added a `customer_text` column, resulting in 21 columns.
- The target is `CSAT Score`, with five observed classes.
- The target is highly imbalanced: class 5 accounts for about 69.40% of records.
- The best reported non-text experiment in the current notes is the price-plus-missingness model, with macro F1 around 0.201495.
- The text model reported macro F1 around 0.2478, but customer remarks may be post-outcome and therefore may leak target information.
- The model results are exploratory and not production-ready.

## 2. Dataset facts

- Raw file: `data/raw/Customer_support_data.csv`.
- Original shape: 85,907 rows × 20 columns.
- Cleaned shape: 85,907 rows × 21 columns.
- No duplicate rows were reported in the initial analysis.
- Customer Remarks missingness: approximately 66.55%.
- Item_price missingness: approximately 79.97%.
- connected_handling_time missingness: approximately 99.72%.
- Train split: 68,725 records.
- Test split: 17,182 records.

### Target distribution

| CSAT class | Count | Percentage (approx.) |
|---:|---:|---:|
| 1 | 11,230 | 13.07% |
| 2 | 1,283 | 1.49% |
| 3 | 2,558 | 2.98% |
| 4 | 11,219 | 13.06% |
| 5 | 59,617 | 69.40% |

## 3. Data preparation

- Cleaning script: `src/clear_dataset.py`.
- Cleaned output: `data/processed/Customer_support_data_cleaned.csv`.
- Text cleaning includes HTML unescaping, tag removal, selected URL/email/contact-string removal, and whitespace normalization.
- The cleaned text column is named `customer_text`.
- Split script: `src/prepare_dataset.py`.
- The split is stratified with `test_size=0.20` and `random_state=42`.
- Raw and processed datasets are excluded by `.gitignore`; retain this protection.

## 4. Feature analysis

- Operational categorical features: `category`, `Sub-category`, `channel_name`, `Tenure Bucket`, and `Agent Shift`.
- Numeric feature investigated: `Item_price`.
- `connected_handling_time` was excluded because almost all values were missing.
- Missingness analysis suggests item-price missingness is systematic and varies across CSAT classes and ticket categories.
- Normalized mutual information values were low for the individual operational features; low univariate association does not prove no combined predictive value.

## 5. Model experiments

### Majority baseline

- Accuracy: approximately 0.6940.
- Macro F1: approximately 0.1639.
- The baseline predicts the majority class and is essential for interpreting imbalanced classification.

### TF-IDF plus operational features

- TF-IDF settings: `max_features=50000`, `ngram_range=(1, 2)`, `min_df=2`, `sublinear_tf=True`.
- Classifier: balanced Logistic Regression, `max_iter=1000`, `random_state=42`.
- Reported accuracy: 0.3491.
- Reported balanced accuracy: 0.2981.
- Reported macro F1: 0.2478.
- Reported weighted F1: 0.4341.
- Important limitation: customer remarks may be post-survey feedback. Verify timestamps and feature availability before treating this as a valid predictive model.

### Operational-only model

- Features: category, Sub-category, channel_name, Tenure Bucket, Agent Shift.
- Reported test accuracy: 0.2955.
- Reported balanced accuracy: 0.2408.
- Reported macro F1: 0.1762.
- Reported weighted F1: 0.3660.
- This result is only modestly above the majority baseline on macro F1.

### Price plus missingness model

- Validation macro F1 values: operational-only 0.1762; operational plus price 0.1823; operational plus price and missingness indicator 0.2015.
- Reported final accuracy: 0.335758.
- Reported balanced accuracy: 0.257746.
- Reported macro precision: 0.236905.
- Reported macro recall: 0.257746.
- Reported macro F1: 0.201495.
- Reported weighted F1: 0.408455.
- Training time: approximately 2.999 seconds.
- Prediction time: approximately 0.040946 seconds for 17,182 records.
- Average prediction time: approximately 2.38 microseconds per record in that run.
- Verify that all compared models use exactly the same split before treating the differences as definitive.

## 6. Evaluation and performance

- Accuracy is strongly influenced by the majority class.
- Macro F1 gives each class equal weight and reveals weak minority-class performance.
- Weighted F1 reflects class support and can hide poor performance on rare classes.
- Balanced accuracy averages recall across classes.
- The price model had very weak precision and F1 for classes 2 and 3.
- The confusion matrix class order is [1, 2, 3, 4, 5].
- The price-model evaluation reconstructed a split from the cleaned CSV; future evaluation should load the persisted `train.csv` and `test.csv` directly.

## 7. Repository and reproducibility

- Repository: `3161-harshit/AI-Powered-Customer-Support-Intelligence-System`.
- Main branch: `main`.
- Source scripts live in `src/`.
- Model artifacts live in `models/`.
- Evaluation outputs live in `evaluation/`.
- Keep data files out of public Git history unless privacy and license checks permit publication.
- Record Python and package versions, input file version, split seed, parameters, and commit hash for each experiment.

## 8. Risks and limitations

- Possible target leakage from customer remarks.
- Strong class imbalance, especially the small supports for CSAT classes 2 and 3.
- High missingness in item price and near-total missingness in handling time.
- Unresolved dataset license and publication permissions must be checked.
- The current model performance is modest and should not be used for automated customer-impacting decisions.
- Operational correlations are not causal explanations.
- A random stratified split may not represent future temporal behavior.

## 9. Future architecture (planned, not claimed as complete)

- Frontend: React.
- Backend: FastAPI / Python.
- ML: Scikit-learn.
- NLP: Hugging Face Transformers.
- RAG: Sentence Transformers and ChromaDB.
- LLM: a cloud API, with secrets stored outside source control.
- Database: PostgreSQL.
- Planned capabilities: ticket classification, sentiment analysis, priority prediction, grounded response suggestions, and analytics.
- Build these only after the prediction task, feature timing, privacy, and evaluation protocol are sound.

## 10. Detailed topic reference

### 10.1 Project purpose
- **Verified context:** Explore customer-support data and predict CSAT scores using Python and classical machine learning.
- **Project interpretation:** The completed work is an exploratory ML study, not a deployed support platform.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Project purpose** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Project purpose** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Project purpose** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Project purpose** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Project purpose** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Project purpose** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Project purpose** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Project purpose** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Project purpose** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Project purpose** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Project purpose** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Project purpose** and record the evidence from the current run.

### 10.2 Dataset source
- **Verified context:** The local raw file is data/raw/Customer_support_data.csv.
- **Project interpretation:** Keep the source file unchanged and record its license and provenance.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Dataset source** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Dataset source** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Dataset source** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Dataset source** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Dataset source** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Dataset source** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Dataset source** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Dataset source** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Dataset source** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Dataset source** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Dataset source** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Dataset source** and record the evidence from the current run.

### 10.3 Dataset size
- **Verified context:** The observed raw dataset contained 85,907 rows and 20 columns.
- **Project interpretation:** Recheck the shape whenever the source data is replaced.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Dataset size** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Dataset size** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Dataset size** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Dataset size** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Dataset size** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Dataset size** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Dataset size** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Dataset size** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Dataset size** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Dataset size** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Dataset size** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Dataset size** and record the evidence from the current run.

### 10.4 Cleaned dataset
- **Verified context:** The cleaning workflow produced 85,907 rows and 21 columns after adding customer_text.
- **Project interpretation:** Confirm the output schema after every cleaning run.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Cleaned dataset** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Cleaned dataset** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Cleaned dataset** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Cleaned dataset** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Cleaned dataset** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Cleaned dataset** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Cleaned dataset** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Cleaned dataset** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Cleaned dataset** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Cleaned dataset** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Cleaned dataset** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Cleaned dataset** and record the evidence from the current run.

### 10.5 Target variable
- **Verified context:** The classification target is CSAT Score with observed classes 1, 2, 3, 4, and 5.
- **Project interpretation:** Do not use the target itself or post-outcome fields as input features.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Target variable** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Target variable** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Target variable** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Target variable** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Target variable** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Target variable** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Target variable** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Target variable** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Target variable** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Target variable** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Target variable** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Target variable** and record the evidence from the current run.

### 10.6 Class imbalance
- **Verified context:** Observed counts were 1: 11,230; 2: 1,283; 3: 2,558; 4: 11,219; 5: 59,617.
- **Project interpretation:** Class 5 represents about 69.40% of records, so accuracy alone is insufficient.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Class imbalance** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Class imbalance** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Class imbalance** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Class imbalance** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Class imbalance** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Class imbalance** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Class imbalance** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Class imbalance** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Class imbalance** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Class imbalance** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Class imbalance** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Class imbalance** and record the evidence from the current run.

### 10.7 Customer remarks missingness
- **Verified context:** Customer Remarks was missing for approximately 66.55% of records.
- **Project interpretation:** Inspect missingness patterns and verify when remarks are collected.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Customer remarks missingness** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Customer remarks missingness** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Customer remarks missingness** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Customer remarks missingness** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Customer remarks missingness** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Customer remarks missingness** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Customer remarks missingness** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Customer remarks missingness** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Customer remarks missingness** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Customer remarks missingness** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Customer remarks missingness** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Customer remarks missingness** and record the evidence from the current run.

### 10.8 Customer text cleaning
- **Verified context:** The cleaning script unescapes HTML, removes tags, strips selected URLs and contact-like strings, and normalizes whitespace.
- **Project interpretation:** Regex cleaning can remove useful context or miss unusual personal-data patterns.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Customer text cleaning** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Customer text cleaning** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Customer text cleaning** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Customer text cleaning** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Customer text cleaning** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Customer text cleaning** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Customer text cleaning** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Customer text cleaning** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Customer text cleaning** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Customer text cleaning** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Customer text cleaning** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Customer text cleaning** and record the evidence from the current run.

### 10.9 Customer text column
- **Verified context:** The cleaned text is stored in customer_text.
- **Project interpretation:** Use customer_text only if it is available at the intended prediction time.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Customer text column** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Customer text column** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Customer text column** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Customer text column** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Customer text column** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Customer text column** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Customer text column** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Customer text column** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Customer text column** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Customer text column** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Customer text column** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Customer text column** and record the evidence from the current run.

### 10.10 Potential target leakage
- **Verified context:** Some remarks look like post-survey feedback, including short ratings such as 'Very good' and 'Very bad'.
- **Project interpretation:** This creates a serious leakage risk for predictions intended before CSAT is submitted.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Potential target leakage** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Potential target leakage** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Potential target leakage** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Potential target leakage** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Potential target leakage** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Potential target leakage** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Potential target leakage** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Potential target leakage** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Potential target leakage** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Potential target leakage** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Potential target leakage** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Potential target leakage** and record the evidence from the current run.

### 10.11 Category feature
- **Verified context:** The category field was included in operational feature experiments.
- **Project interpretation:** Verify that category is assigned before the prediction point.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Category feature** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Category feature** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Category feature** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Category feature** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Category feature** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Category feature** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Category feature** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Category feature** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Category feature** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Category feature** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Category feature** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Category feature** and record the evidence from the current run.

### 10.12 Sub-category feature
- **Verified context:** Sub-category was included in operational feature experiments.
- **Project interpretation:** Monitor spelling variants, new labels, and category taxonomy changes.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Sub-category feature** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Sub-category feature** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Sub-category feature** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Sub-category feature** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Sub-category feature** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Sub-category feature** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Sub-category feature** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Sub-category feature** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Sub-category feature** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Sub-category feature** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Sub-category feature** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Sub-category feature** and record the evidence from the current run.

### 10.13 Channel feature
- **Verified context:** Observed channel counts were Inbound 68,142; Outcall 14,742; Email 3,023.
- **Project interpretation:** Channel may reflect workflow and customer mix; it is not necessarily causal.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Channel feature** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Channel feature** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Channel feature** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Channel feature** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Channel feature** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Channel feature** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Channel feature** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Channel feature** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Channel feature** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Channel feature** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Channel feature** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Channel feature** and record the evidence from the current run.

### 10.14 Tenure Bucket feature
- **Verified context:** Tenure Bucket was included as an operational categorical feature.
- **Project interpretation:** Confirm the definition and whether it is known at prediction time.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Tenure Bucket feature** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Tenure Bucket feature** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Tenure Bucket feature** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Tenure Bucket feature** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Tenure Bucket feature** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Tenure Bucket feature** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Tenure Bucket feature** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Tenure Bucket feature** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Tenure Bucket feature** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Tenure Bucket feature** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Tenure Bucket feature** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Tenure Bucket feature** and record the evidence from the current run.

### 10.15 Agent Shift feature
- **Verified context:** Agent Shift was included as an operational categorical feature.
- **Project interpretation:** Avoid using this feature to make unsupported judgments about individual agents.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Agent Shift feature** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Agent Shift feature** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Agent Shift feature** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Agent Shift feature** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Agent Shift feature** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Agent Shift feature** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Agent Shift feature** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Agent Shift feature** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Agent Shift feature** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Agent Shift feature** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Agent Shift feature** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Agent Shift feature** and record the evidence from the current run.

### 10.16 Item price
- **Verified context:** Item_price was available for 17,206 records and missing for 68,701 records, about 79.97% missing.
- **Project interpretation:** Its missingness may encode workflow or product-category information.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Item price** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Item price** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Item price** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Item price** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Item price** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Item price** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Item price** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Item price** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Item price** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Item price** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Item price** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Item price** and record the evidence from the current run.

### 10.17 Item price distribution
- **Verified context:** Observed item-price mean was about 5,660.77 and median about 979.
- **Project interpretation:** The distribution is highly skewed; inspect quantiles and outliers.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Item price distribution** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Item price distribution** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Item price distribution** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Item price distribution** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Item price distribution** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Item price distribution** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Item price distribution** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Item price distribution** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Item price distribution** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Item price distribution** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Item price distribution** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Item price distribution** and record the evidence from the current run.

### 10.18 Handling time
- **Verified context:** connected_handling_time was available for only 242 records and missing for about 99.72%.
- **Project interpretation:** It was excluded from current modeling because coverage is extremely low.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Handling time** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Handling time** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Handling time** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Handling time** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Handling time** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Handling time** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Handling time** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Handling time** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Handling time** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Handling time** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Handling time** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Handling time** and record the evidence from the current run.

### 10.19 Train-test split
- **Verified context:** The prepared split used 80% training and 20% testing with stratification and random_state=42.
- **Project interpretation:** Keep the final test set untouched during feature selection and tuning.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Train-test split** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Train-test split** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Train-test split** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Train-test split** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Train-test split** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Train-test split** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Train-test split** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Train-test split** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Train-test split** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Train-test split** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Train-test split** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Train-test split** and record the evidence from the current run.

### 10.20 Training size
- **Verified context:** The reported training split contained 68,725 records.
- **Project interpretation:** Verify counts against the saved train.csv after any data refresh.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Training size** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Training size** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Training size** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Training size** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Training size** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Training size** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Training size** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Training size** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Training size** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Training size** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Training size** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Training size** and record the evidence from the current run.

### 10.21 Test size
- **Verified context:** The reported test split contained 17,182 records.
- **Project interpretation:** Use the same frozen test records for final model comparisons.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Test size** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Test size** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Test size** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Test size** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Test size** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Test size** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Test size** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Test size** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Test size** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Test size** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Test size** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Test size** and record the evidence from the current run.

### 10.22 Majority baseline
- **Verified context:** The majority baseline accuracy was about 0.6940 and macro F1 about 0.1639.
- **Project interpretation:** A baseline is essential because the target is strongly imbalanced.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Majority baseline** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Majority baseline** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Majority baseline** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Majority baseline** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Majority baseline** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Majority baseline** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Majority baseline** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Majority baseline** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Majority baseline** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Majority baseline** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Majority baseline** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Majority baseline** and record the evidence from the current run.

### 10.23 TF-IDF settings
- **Verified context:** The text model used max_features=50,000, ngram_range=(1,2), min_df=2, and sublinear_tf=True.
- **Project interpretation:** Fit TF-IDF only on training text through a pipeline.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **TF-IDF settings** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **TF-IDF settings** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **TF-IDF settings** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **TF-IDF settings** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **TF-IDF settings** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **TF-IDF settings** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **TF-IDF settings** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **TF-IDF settings** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **TF-IDF settings** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **TF-IDF settings** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **TF-IDF settings** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **TF-IDF settings** and record the evidence from the current run.

### 10.24 Text model
- **Verified context:** The first model combined TF-IDF text and operational categorical features with balanced Logistic Regression.
- **Project interpretation:** Treat its result as exploratory until remark timing is verified.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Text model** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Text model** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Text model** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Text model** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Text model** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Text model** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Text model** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Text model** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Text model** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Text model** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Text model** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Text model** and record the evidence from the current run.

### 10.25 Operational model
- **Verified context:** The operational-only model used category, Sub-category, channel_name, Tenure Bucket, and Agent Shift.
- **Project interpretation:** This is a more conservative feature set than one using customer remarks.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Operational model** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Operational model** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Operational model** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Operational model** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Operational model** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Operational model** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Operational model** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Operational model** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Operational model** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Operational model** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Operational model** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Operational model** and record the evidence from the current run.

### 10.26 Logistic Regression
- **Verified context:** Logistic Regression was used as a transparent baseline classifier.
- **Project interpretation:** Record solver, regularization, class weights, iteration limit, and random seed.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Logistic Regression** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Logistic Regression** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Logistic Regression** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Logistic Regression** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Logistic Regression** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Logistic Regression** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Logistic Regression** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Logistic Regression** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Logistic Regression** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Logistic Regression** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Logistic Regression** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Logistic Regression** and record the evidence from the current run.

### 10.27 Class weighting
- **Verified context:** Balanced class weights were used in selected experiments.
- **Project interpretation:** Class weighting can improve minority recall while reducing overall accuracy.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Class weighting** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Class weighting** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Class weighting** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Class weighting** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Class weighting** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Class weighting** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Class weighting** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Class weighting** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Class weighting** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Class weighting** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Class weighting** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Class weighting** and record the evidence from the current run.

### 10.28 Operational test accuracy
- **Verified context:** The operational-only model reported test accuracy around 0.2955.
- **Project interpretation:** Interpret accuracy beside the 0.6940 majority baseline and class metrics.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Operational test accuracy** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Operational test accuracy** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Operational test accuracy** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Operational test accuracy** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Operational test accuracy** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Operational test accuracy** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Operational test accuracy** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Operational test accuracy** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Operational test accuracy** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Operational test accuracy** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Operational test accuracy** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Operational test accuracy** and record the evidence from the current run.

### 10.29 Operational balanced accuracy
- **Verified context:** The operational-only model reported balanced accuracy around 0.2408.
- **Project interpretation:** Balanced accuracy averages recall across classes.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Operational balanced accuracy** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Operational balanced accuracy** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Operational balanced accuracy** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Operational balanced accuracy** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Operational balanced accuracy** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Operational balanced accuracy** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Operational balanced accuracy** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Operational balanced accuracy** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Operational balanced accuracy** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Operational balanced accuracy** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Operational balanced accuracy** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Operational balanced accuracy** and record the evidence from the current run.

### 10.30 Operational macro F1
- **Verified context:** The operational-only model reported macro F1 around 0.1762.
- **Project interpretation:** This was only modestly above the majority baseline macro F1 of 0.1639.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Operational macro F1** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Operational macro F1** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Operational macro F1** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Operational macro F1** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Operational macro F1** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Operational macro F1** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Operational macro F1** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Operational macro F1** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Operational macro F1** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Operational macro F1** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Operational macro F1** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Operational macro F1** and record the evidence from the current run.

### 10.31 Operational weighted F1
- **Verified context:** The operational-only model reported weighted F1 around 0.3660.
- **Project interpretation:** Weighted F1 is influenced strongly by the majority class.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Operational weighted F1** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Operational weighted F1** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Operational weighted F1** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Operational weighted F1** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Operational weighted F1** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Operational weighted F1** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Operational weighted F1** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Operational weighted F1** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Operational weighted F1** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Operational weighted F1** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Operational weighted F1** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Operational weighted F1** and record the evidence from the current run.

### 10.32 Price experiment
- **Verified context:** Validation macro F1 values were approximately 0.1762 operational-only, 0.1823 with price, and 0.2015 with price plus a missingness indicator.
- **Project interpretation:** Confirm all variants use identical split indices before comparing gains.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Price experiment** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Price experiment** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Price experiment** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Price experiment** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Price experiment** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Price experiment** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Price experiment** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Price experiment** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Price experiment** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Price experiment** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Price experiment** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Price experiment** and record the evidence from the current run.

### 10.33 Price model accuracy
- **Verified context:** The price-plus-missingness model reported test accuracy around 0.335758.
- **Project interpretation:** This is below the majority baseline accuracy, despite better macro F1.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Price model accuracy** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Price model accuracy** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Price model accuracy** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Price model accuracy** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Price model accuracy** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Price model accuracy** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Price model accuracy** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Price model accuracy** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Price model accuracy** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Price model accuracy** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Price model accuracy** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Price model accuracy** and record the evidence from the current run.

### 10.34 Price model balanced accuracy
- **Verified context:** The price-plus-missingness model reported balanced accuracy around 0.257746.
- **Project interpretation:** This is an improvement over the operational-only result but remains limited.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Price model balanced accuracy** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Price model balanced accuracy** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Price model balanced accuracy** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Price model balanced accuracy** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Price model balanced accuracy** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Price model balanced accuracy** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Price model balanced accuracy** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Price model balanced accuracy** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Price model balanced accuracy** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Price model balanced accuracy** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Price model balanced accuracy** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Price model balanced accuracy** and record the evidence from the current run.

### 10.35 Price model macro F1
- **Verified context:** The price-plus-missingness model reported macro F1 around 0.201495.
- **Project interpretation:** This is about 14.35% higher relative to the operational-only macro F1 of 0.1762.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Price model macro F1** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Price model macro F1** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Price model macro F1** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Price model macro F1** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Price model macro F1** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Price model macro F1** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Price model macro F1** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Price model macro F1** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Price model macro F1** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Price model macro F1** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Price model macro F1** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Price model macro F1** and record the evidence from the current run.

### 10.36 Price model weighted F1
- **Verified context:** The price-plus-missingness model reported weighted F1 around 0.408455.
- **Project interpretation:** Always report the class-level results alongside this aggregate.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Price model weighted F1** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Price model weighted F1** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Price model weighted F1** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Price model weighted F1** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Price model weighted F1** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Price model weighted F1** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Price model weighted F1** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Price model weighted F1** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Price model weighted F1** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Price model weighted F1** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Price model weighted F1** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Price model weighted F1** and record the evidence from the current run.

### 10.37 Price model training time
- **Verified context:** The reported training time was about 2.999 seconds on the user's machine.
- **Project interpretation:** Runtime depends on hardware, software versions, and system load.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Price model training time** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Price model training time** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Price model training time** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Price model training time** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Price model training time** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Price model training time** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Price model training time** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Price model training time** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Price model training time** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Price model training time** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Price model training time** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Price model training time** and record the evidence from the current run.

### 10.38 Price model prediction time
- **Verified context:** The reported prediction time was about 0.040946 seconds for 17,182 test records.
- **Project interpretation:** Document whether model loading and preprocessing are included in timing.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Price model prediction time** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Price model prediction time** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Price model prediction time** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Price model prediction time** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Price model prediction time** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Price model prediction time** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Price model prediction time** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Price model prediction time** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Price model prediction time** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Price model prediction time** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Price model prediction time** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Price model prediction time** and record the evidence from the current run.

### 10.39 Average prediction time
- **Verified context:** The reported average was about 2.38 microseconds per record for that batch.
- **Project interpretation:** Do not generalize this figure to other hardware or deployment conditions.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Average prediction time** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Average prediction time** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Average prediction time** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Average prediction time** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Average prediction time** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Average prediction time** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Average prediction time** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Average prediction time** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Average prediction time** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Average prediction time** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Average prediction time** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Average prediction time** and record the evidence from the current run.

### 10.40 Minority class performance
- **Verified context:** Classes 2 and 3 had very low precision and F1 in the reported price-model classification report.
- **Project interpretation:** Inspect confusion patterns before claiming the model is useful for every rating class.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Minority class performance** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Minority class performance** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Minority class performance** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Minority class performance** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Minority class performance** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Minority class performance** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Minority class performance** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Minority class performance** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Minority class performance** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Minority class performance** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Minority class performance** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Minority class performance** and record the evidence from the current run.

### 10.41 Confusion matrix
- **Verified context:** The price-model matrix used class order [1, 2, 3, 4, 5].
- **Project interpretation:** Rows are true classes and columns are predicted classes under sklearn's standard convention.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Confusion matrix** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Confusion matrix** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Confusion matrix** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Confusion matrix** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Confusion matrix** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Confusion matrix** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Confusion matrix** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Confusion matrix** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Confusion matrix** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Confusion matrix** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Confusion matrix** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Confusion matrix** and record the evidence from the current run.

### 10.42 NMI analysis
- **Verified context:** Reported normalized mutual information was low: Sub-category 0.0099, category 0.0041, channel 0.0012, Tenure Bucket 0.0012, Agent Shift 0.0011.
- **Project interpretation:** NMI is an association measure, not a causal explanation.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **NMI analysis** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **NMI analysis** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **NMI analysis** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **NMI analysis** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **NMI analysis** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **NMI analysis** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **NMI analysis** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **NMI analysis** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **NMI analysis** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **NMI analysis** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **NMI analysis** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **NMI analysis** and record the evidence from the current run.

### 10.43 Missingness by CSAT
- **Verified context:** Item-price missingness varied across CSAT classes, from about 70.01% for class 1 to 81.94% for class 5.
- **Project interpretation:** Check whether this pattern is stable across future data.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Missingness by CSAT** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Missingness by CSAT** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Missingness by CSAT** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Missingness by CSAT** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Missingness by CSAT** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Missingness by CSAT** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Missingness by CSAT** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Missingness by CSAT** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Missingness by CSAT** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Missingness by CSAT** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Missingness by CSAT** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Missingness by CSAT** and record the evidence from the current run.

### 10.44 Missingness by category
- **Verified context:** Item-price missingness was especially high for several categories, including Product Queries and Shopzilla Related.
- **Project interpretation:** Missingness can reflect data-collection rules and workflow differences.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Missingness by category** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Missingness by category** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Missingness by category** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Missingness by category** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Missingness by category** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Missingness by category** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Missingness by category** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Missingness by category** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Missingness by category** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Missingness by category** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Missingness by category** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Missingness by category** and record the evidence from the current run.

### 10.45 Data privacy
- **Verified context:** Customer-support datasets can contain personal, transactional, or operationally sensitive information.
- **Project interpretation:** Do not publish raw or processed customer records without privacy and license approval.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Data privacy** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Data privacy** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Data privacy** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Data privacy** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Data privacy** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Data privacy** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Data privacy** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Data privacy** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Data privacy** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Data privacy** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Data privacy** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Data privacy** and record the evidence from the current run.

### 10.46 Git ignore
- **Verified context:** The project .gitignore excludes data/raw/ and data/processed/.
- **Project interpretation:** An ignore rule does not remove a file already tracked by Git.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Git ignore** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Git ignore** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Git ignore** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Git ignore** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Git ignore** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Git ignore** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Git ignore** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Git ignore** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Git ignore** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Git ignore** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Git ignore** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Git ignore** and record the evidence from the current run.

### 10.47 Repository layout
- **Verified context:** The repository separates data, src, models, evaluation, backend, and frontend areas.
- **Project interpretation:** Keep scripts, metrics, and documentation organized by responsibility.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Repository layout** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Repository layout** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Repository layout** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Repository layout** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Repository layout** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Repository layout** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Repository layout** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Repository layout** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Repository layout** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Repository layout** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Repository layout** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Repository layout** and record the evidence from the current run.

### 10.48 Cleaning script
- **Verified context:** The cleaning script is src/clear_dataset.py.
- **Project interpretation:** Run it from the project root and check the resulting schema and null counts.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Cleaning script** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Cleaning script** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Cleaning script** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Cleaning script** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Cleaning script** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Cleaning script** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Cleaning script** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Cleaning script** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Cleaning script** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Cleaning script** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Cleaning script** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Cleaning script** and record the evidence from the current run.

### 10.49 Feature analysis script
- **Verified context:** The exploratory feature script is src/analyze_features.py.
- **Project interpretation:** Keep analysis output reproducible and tied to a dataset version.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Feature analysis script** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Feature analysis script** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Feature analysis script** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Feature analysis script** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Feature analysis script** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Feature analysis script** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Feature analysis script** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Feature analysis script** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Feature analysis script** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Feature analysis script** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Feature analysis script** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Feature analysis script** and record the evidence from the current run.

### 10.50 Dataset preparation script
- **Verified context:** The split preparation script is src/prepare_dataset.py.
- **Project interpretation:** Persist the split and document the seed and target stratification.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Dataset preparation script** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Dataset preparation script** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Dataset preparation script** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Dataset preparation script** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Dataset preparation script** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Dataset preparation script** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Dataset preparation script** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Dataset preparation script** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Dataset preparation script** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Dataset preparation script** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Dataset preparation script** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Dataset preparation script** and record the evidence from the current run.

### 10.51 Initial model script
- **Verified context:** The first model training script is src/train_model.py.
- **Project interpretation:** Ensure preprocessing is fitted on training data only.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Initial model script** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Initial model script** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Initial model script** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Initial model script** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Initial model script** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Initial model script** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Initial model script** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Initial model script** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Initial model script** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Initial model script** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Initial model script** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Initial model script** and record the evidence from the current run.

### 10.52 Operational model script
- **Verified context:** The operational training script is src/train_operational_model.py.
- **Project interpretation:** Record validation selection criteria and final refit procedure.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Operational model script** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Operational model script** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Operational model script** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Operational model script** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Operational model script** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Operational model script** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Operational model script** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Operational model script** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Operational model script** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Operational model script** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Operational model script** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Operational model script** and record the evidence from the current run.

### 10.53 Operational evaluation script
- **Verified context:** The operational evaluation script is src/evaluate_operational_model.py.
- **Project interpretation:** Evaluate against the saved test records.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Operational evaluation script** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Operational evaluation script** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Operational evaluation script** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Operational evaluation script** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Operational evaluation script** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Operational evaluation script** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Operational evaluation script** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Operational evaluation script** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Operational evaluation script** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Operational evaluation script** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Operational evaluation script** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Operational evaluation script** and record the evidence from the current run.

### 10.54 Numeric analysis script
- **Verified context:** The numeric analysis script is src/analyze_numeric_features.py.
- **Project interpretation:** Recheck availability counts and summary statistics after data changes.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Numeric analysis script** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Numeric analysis script** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Numeric analysis script** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Numeric analysis script** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Numeric analysis script** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Numeric analysis script** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Numeric analysis script** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Numeric analysis script** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Numeric analysis script** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Numeric analysis script** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Numeric analysis script** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Numeric analysis script** and record the evidence from the current run.

### 10.55 Missingness analysis script
- **Verified context:** The missingness script is src/analyze_missingness.py.
- **Project interpretation:** Report missingness overall and by relevant groups.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Missingness analysis script** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Missingness analysis script** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Missingness analysis script** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Missingness analysis script** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Missingness analysis script** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Missingness analysis script** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Missingness analysis script** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Missingness analysis script** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Missingness analysis script** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Missingness analysis script** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Missingness analysis script** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Missingness analysis script** and record the evidence from the current run.

### 10.56 Price comparison script
- **Verified context:** The price experiment script is src/compare_price_features.py.
- **Project interpretation:** Use the same validation records for every feature variant.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Price comparison script** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Price comparison script** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Price comparison script** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Price comparison script** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Price comparison script** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Price comparison script** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Price comparison script** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Price comparison script** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Price comparison script** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Price comparison script** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Price comparison script** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Price comparison script** and record the evidence from the current run.

### 10.57 Price evaluation script
- **Verified context:** The final price evaluation script is src/evaluate_price_model.py.
- **Project interpretation:** Prefer loading persisted train.csv and test.csv instead of reconstructing indices.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Price evaluation script** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Price evaluation script** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Price evaluation script** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Price evaluation script** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Price evaluation script** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Price evaluation script** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Price evaluation script** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Price evaluation script** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Price evaluation script** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Price evaluation script** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Price evaluation script** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Price evaluation script** and record the evidence from the current run.

### 10.58 Model artifact
- **Verified context:** The project saved joblib model files under models/.
- **Project interpretation:** Only load trusted artifacts and record package versions.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Model artifact** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Model artifact** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Model artifact** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Model artifact** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Model artifact** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Model artifact** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Model artifact** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Model artifact** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Model artifact** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Model artifact** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Model artifact** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Model artifact** and record the evidence from the current run.

### 10.59 Metrics artifact
- **Verified context:** Evaluation metrics were saved as JSON under evaluation/.
- **Project interpretation:** Keep JSON generated by scripts rather than editing metric values manually.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Metrics artifact** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Metrics artifact** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Metrics artifact** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Metrics artifact** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Metrics artifact** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Metrics artifact** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Metrics artifact** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Metrics artifact** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Metrics artifact** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Metrics artifact** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Metrics artifact** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Metrics artifact** and record the evidence from the current run.

### 10.60 Classification report
- **Verified context:** Classification reports were saved as text files under evaluation/.
- **Project interpretation:** Check class order and support counts when comparing reports.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Classification report** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Classification report** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Classification report** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Classification report** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Classification report** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Classification report** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Classification report** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Classification report** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Classification report** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Classification report** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Classification report** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Classification report** and record the evidence from the current run.

### 10.61 README
- **Verified context:** README.md is the repository's main entry point.
- **Project interpretation:** Keep the README concise and link to this detailed research note.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **README** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **README** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **README** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **README** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **README** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **README** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **README** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **README** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **README** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **README** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **README** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **README** and record the evidence from the current run.

### 10.62 Research note
- **Verified context:** notes.md is intended as a detailed project research and experiment record.
- **Project interpretation:** Update it when implementation or measured results change.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Research note** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Research note** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Research note** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Research note** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Research note** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Research note** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Research note** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Research note** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Research note** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Research note** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Research note** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Research note** and record the evidence from the current run.

### 10.63 Future React frontend
- **Verified context:** React is part of the proposed future application architecture.
- **Project interpretation:** Do not describe the frontend as completed until it is implemented and tested.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Future React frontend** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Future React frontend** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Future React frontend** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Future React frontend** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Future React frontend** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Future React frontend** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Future React frontend** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Future React frontend** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Future React frontend** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Future React frontend** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Future React frontend** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Future React frontend** and record the evidence from the current run.

### 10.64 Future FastAPI backend
- **Verified context:** FastAPI is part of the proposed future backend architecture.
- **Project interpretation:** A local training script is not equivalent to a deployed API.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Future FastAPI backend** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Future FastAPI backend** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Future FastAPI backend** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Future FastAPI backend** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Future FastAPI backend** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Future FastAPI backend** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Future FastAPI backend** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Future FastAPI backend** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Future FastAPI backend** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Future FastAPI backend** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Future FastAPI backend** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Future FastAPI backend** and record the evidence from the current run.

### 10.65 Future RAG
- **Verified context:** RAG is proposed for retrieving approved support knowledge and grounding response suggestions.
- **Project interpretation:** RAG is not part of the completed CSAT model experiments described here.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Future RAG** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Future RAG** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Future RAG** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Future RAG** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Future RAG** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Future RAG** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Future RAG** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Future RAG** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Future RAG** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Future RAG** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Future RAG** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Future RAG** and record the evidence from the current run.

### 10.66 Future ChromaDB
- **Verified context:** ChromaDB is proposed as a vector store for a future knowledge-base retrieval workflow.
- **Project interpretation:** A vector database alone does not guarantee answer accuracy.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Future ChromaDB** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Future ChromaDB** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Future ChromaDB** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Future ChromaDB** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Future ChromaDB** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Future ChromaDB** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Future ChromaDB** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Future ChromaDB** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Future ChromaDB** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Future ChromaDB** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Future ChromaDB** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Future ChromaDB** and record the evidence from the current run.

### 10.67 Future embeddings
- **Verified context:** Sentence Transformers is proposed for semantic embeddings in the future RAG component.
- **Project interpretation:** Record model name, version, vector dimension, and normalization settings.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Future embeddings** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Future embeddings** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Future embeddings** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Future embeddings** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Future embeddings** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Future embeddings** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Future embeddings** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Future embeddings** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Future embeddings** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Future embeddings** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Future embeddings** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Future embeddings** and record the evidence from the current run.

### 10.68 Future LLM
- **Verified context:** A cloud LLM API is proposed for response generation.
- **Project interpretation:** Keep API keys secret and ground outputs in approved knowledge sources.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Future LLM** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Future LLM** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Future LLM** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Future LLM** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Future LLM** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Future LLM** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Future LLM** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Future LLM** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Future LLM** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Future LLM** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Future LLM** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Future LLM** and record the evidence from the current run.

### 10.69 Future PostgreSQL
- **Verified context:** PostgreSQL is proposed for application records and structured data.
- **Project interpretation:** Design access controls, retention rules, migrations, and backups before storing customer data.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Future PostgreSQL** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Future PostgreSQL** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Future PostgreSQL** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Future PostgreSQL** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Future PostgreSQL** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Future PostgreSQL** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Future PostgreSQL** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Future PostgreSQL** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Future PostgreSQL** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Future PostgreSQL** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Future PostgreSQL** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Future PostgreSQL** and record the evidence from the current run.

### 10.70 Ticket classification roadmap
- **Verified context:** Ticket classification is a planned capability distinct from the current CSAT prediction task.
- **Project interpretation:** Create a separate labeled target and evaluation plan.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Ticket classification roadmap** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Ticket classification roadmap** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Ticket classification roadmap** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Ticket classification roadmap** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Ticket classification roadmap** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Ticket classification roadmap** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Ticket classification roadmap** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Ticket classification roadmap** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Ticket classification roadmap** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Ticket classification roadmap** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Ticket classification roadmap** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Ticket classification roadmap** and record the evidence from the current run.

### 10.71 Sentiment analysis roadmap
- **Verified context:** Sentiment analysis is a proposed NLP capability for customer messages.
- **Project interpretation:** Sentiment and CSAT are related but are not interchangeable labels.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Sentiment analysis roadmap** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Sentiment analysis roadmap** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Sentiment analysis roadmap** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Sentiment analysis roadmap** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Sentiment analysis roadmap** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Sentiment analysis roadmap** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Sentiment analysis roadmap** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Sentiment analysis roadmap** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Sentiment analysis roadmap** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Sentiment analysis roadmap** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Sentiment analysis roadmap** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Sentiment analysis roadmap** and record the evidence from the current run.

### 10.72 Priority prediction roadmap
- **Verified context:** Priority prediction is a proposed capability for estimating ticket urgency.
- **Project interpretation:** Define urgency labels and the costs of false negatives and false positives.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Priority prediction roadmap** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Priority prediction roadmap** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Priority prediction roadmap** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Priority prediction roadmap** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Priority prediction roadmap** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Priority prediction roadmap** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Priority prediction roadmap** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Priority prediction roadmap** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Priority prediction roadmap** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Priority prediction roadmap** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Priority prediction roadmap** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Priority prediction roadmap** and record the evidence from the current run.

### 10.73 Response suggestions roadmap
- **Verified context:** AI response suggestions are planned to assist support agents.
- **Project interpretation:** Keep a human review step and provide source references where possible.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Response suggestions roadmap** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Response suggestions roadmap** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Response suggestions roadmap** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Response suggestions roadmap** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Response suggestions roadmap** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Response suggestions roadmap** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Response suggestions roadmap** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Response suggestions roadmap** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Response suggestions roadmap** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Response suggestions roadmap** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Response suggestions roadmap** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Response suggestions roadmap** and record the evidence from the current run.

### 10.74 Dashboard roadmap
- **Verified context:** An analytics dashboard is part of the future product concept.
- **Project interpretation:** Display uncertainty, class-level performance, and data freshness clearly.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Dashboard roadmap** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Dashboard roadmap** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Dashboard roadmap** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Dashboard roadmap** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Dashboard roadmap** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Dashboard roadmap** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Dashboard roadmap** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Dashboard roadmap** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Dashboard roadmap** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Dashboard roadmap** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Dashboard roadmap** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Dashboard roadmap** and record the evidence from the current run.

### 10.75 Model monitoring
- **Verified context:** A future deployed model would need input drift and delayed outcome monitoring.
- **Project interpretation:** Track feature distributions, missingness, and class-level performance over time.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Model monitoring** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Model monitoring** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Model monitoring** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Model monitoring** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Model monitoring** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Model monitoring** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Model monitoring** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Model monitoring** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Model monitoring** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Model monitoring** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Model monitoring** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Model monitoring** and record the evidence from the current run.

### 10.76 Time-based validation
- **Verified context:** The dataset includes timestamp fields with substantial missingness in some columns.
- **Project interpretation:** Consider chronological validation if the intended use is prediction on future tickets.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Time-based validation** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Time-based validation** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Time-based validation** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Time-based validation** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Time-based validation** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Time-based validation** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Time-based validation** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Time-based validation** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Time-based validation** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Time-based validation** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Time-based validation** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Time-based validation** and record the evidence from the current run.

### 10.77 Feature availability
- **Verified context:** Each feature needs a clear definition of when it becomes available in the workflow.
- **Project interpretation:** Exclude fields recorded after the intended prediction point.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Feature availability** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Feature availability** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Feature availability** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Feature availability** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Feature availability** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Feature availability** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Feature availability** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Feature availability** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Feature availability** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Feature availability** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Feature availability** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Feature availability** and record the evidence from the current run.

### 10.78 Data quality
- **Verified context:** Data checks should cover schema, missing values, duplicate records, invalid targets, and malformed timestamps.
- **Project interpretation:** Fail early if required fields are missing or unexpected target labels appear.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Data quality** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Data quality** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Data quality** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Data quality** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Data quality** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Data quality** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Data quality** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Data quality** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Data quality** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Data quality** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Data quality** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Data quality** and record the evidence from the current run.

### 10.79 Automated tests
- **Verified context:** Tests should cover cleaning, schema validation, preprocessing, model loading, and prediction.
- **Project interpretation:** Use synthetic fixtures instead of real customer records.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Automated tests** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Automated tests** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Automated tests** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Automated tests** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Automated tests** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Automated tests** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Automated tests** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Automated tests** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Automated tests** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Automated tests** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Automated tests** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Automated tests** and record the evidence from the current run.

### 10.80 Inference contract
- **Verified context:** A future prediction endpoint must receive the feature schema expected by the fitted pipeline.
- **Project interpretation:** Test missing fields, unknown categories, empty text, and invalid numeric values.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Inference contract** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Inference contract** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Inference contract** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Inference contract** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Inference contract** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Inference contract** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Inference contract** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Inference contract** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Inference contract** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Inference contract** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Inference contract** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Inference contract** and record the evidence from the current run.

### 10.81 Model calibration
- **Verified context:** If predicted probabilities are exposed, calibration should be checked.
- **Project interpretation:** Do not assume class-weighted Logistic Regression probabilities are calibrated.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Model calibration** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Model calibration** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Model calibration** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Model calibration** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Model calibration** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Model calibration** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Model calibration** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Model calibration** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Model calibration** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Model calibration** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Model calibration** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Model calibration** and record the evidence from the current run.

### 10.82 Error analysis
- **Verified context:** Misclassifications should be reviewed by true class, predicted class, category, and missingness pattern.
- **Project interpretation:** Use privacy-safe aggregates or synthetic examples in public reports.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Error analysis** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Error analysis** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Error analysis** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Error analysis** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Error analysis** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Error analysis** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Error analysis** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Error analysis** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Error analysis** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Error analysis** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Error analysis** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Error analysis** and record the evidence from the current run.

### 10.83 Fairness review
- **Verified context:** If future use affects customers or employees, assess error differences across appropriate groups.
- **Project interpretation:** Do not infer individual performance or causation from incomplete operational data.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Fairness review** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Fairness review** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Fairness review** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Fairness review** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Fairness review** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Fairness review** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Fairness review** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Fairness review** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Fairness review** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Fairness review** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Fairness review** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Fairness review** and record the evidence from the current run.

### 10.84 Human oversight
- **Verified context:** Future AI-generated replies and priority suggestions should remain reviewable by support staff.
- **Project interpretation:** Provide the ability to edit, reject, or escalate suggestions.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Human oversight** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Human oversight** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Human oversight** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Human oversight** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Human oversight** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Human oversight** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Human oversight** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Human oversight** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Human oversight** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Human oversight** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Human oversight** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Human oversight** and record the evidence from the current run.

### 10.85 Security
- **Verified context:** Secrets and credentials must not be committed to source control.
- **Project interpretation:** Use environment variables and rotate any credential accidentally exposed.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Security** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Security** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Security** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Security** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Security** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Security** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Security** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Security** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Security** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Security** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Security** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Security** and record the evidence from the current run.

### 10.86 Dependency management
- **Verified context:** The project should record Python and package versions.
- **Project interpretation:** Recreate the environment from a clean install before release.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Dependency management** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Dependency management** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Dependency management** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Dependency management** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Dependency management** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Dependency management** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Dependency management** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Dependency management** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Dependency management** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Dependency management** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Dependency management** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Dependency management** and record the evidence from the current run.

### 10.87 Windows paths
- **Verified context:** Development has used Windows and a local project folder on drive L:.
- **Project interpretation:** Use pathlib and configurable paths rather than hard-coded personal directories.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Windows paths** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Windows paths** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Windows paths** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Windows paths** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Windows paths** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Windows paths** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Windows paths** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Windows paths** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Windows paths** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Windows paths** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Windows paths** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Windows paths** and record the evidence from the current run.

### 10.88 Git workflow
- **Verified context:** The main branch is used for the project repository.
- **Project interpretation:** Fetch, inspect status, stage intended files, commit clearly, and push.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Git workflow** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Git workflow** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Git workflow** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Git workflow** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Git workflow** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Git workflow** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Git workflow** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Git workflow** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Git workflow** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Git workflow** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Git workflow** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Git workflow** and record the evidence from the current run.

### 10.89 Git verification
- **Verified context:** Verify remote branch history and file contents after important documentation changes.
- **Project interpretation:** Do not claim a change is on GitHub until the remote content is checked.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Git verification** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Git verification** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Git verification** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Git verification** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Git verification** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Git verification** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Git verification** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Git verification** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Git verification** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Git verification** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Git verification** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Git verification** and record the evidence from the current run.

### 10.90 Resume accuracy
- **Verified context:** The current project demonstrates data cleaning, feature analysis, model comparison, and evaluation.
- **Project interpretation:** Do not claim production deployment or strong predictive performance.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Resume accuracy** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Resume accuracy** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Resume accuracy** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Resume accuracy** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Resume accuracy** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Resume accuracy** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Resume accuracy** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Resume accuracy** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Resume accuracy** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Resume accuracy** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Resume accuracy** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Resume accuracy** and record the evidence from the current run.

### 10.91 Project limitations
- **Verified context:** The current results are modest and minority classes are difficult to predict.
- **Project interpretation:** Present limitations honestly and use them to motivate further experiments.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Project limitations** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Project limitations** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Project limitations** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Project limitations** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Project limitations** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Project limitations** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Project limitations** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Project limitations** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Project limitations** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Project limitations** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Project limitations** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Project limitations** and record the evidence from the current run.

### 10.92 Next research step
- **Verified context:** The next priority is to verify feature timing and remove any possible target leakage.
- **Project interpretation:** Reliable problem framing should come before adding more complex models.
- **Objective and context:** Define the precise question, target user, prediction moment, and intended decision. Apply this specifically to **Next research step** and record the evidence from the current run.
- **Data requirements:** Document source, schema, data types, units, missingness, and access restrictions. Apply this specifically to **Next research step** and record the evidence from the current run.
- **Implementation notes:** Describe the script, pipeline, transformations, parameters, and artifact paths. Apply this specifically to **Next research step** and record the evidence from the current run.
- **Validation checklist:** Specify input checks, expected output, reproducibility checks, and edge cases. Apply this specifically to **Next research step** and record the evidence from the current run.
- **Evaluation interpretation:** Explain the metric or observation without overstating what it proves. Apply this specifically to **Next research step** and record the evidence from the current run.
- **Risks and limitations:** Record leakage, bias, drift, privacy, data quality, and operational constraints. Apply this specifically to **Next research step** and record the evidence from the current run.
- **Reproducibility record:** Record dataset version, random seed, package versions, machine details, and commit. Apply this specifically to **Next research step** and record the evidence from the current run.
- **Operational considerations:** Describe how the feature or result would be used, monitored, and reviewed. Apply this specifically to **Next research step** and record the evidence from the current run.
- **Documentation standard:** Separate verified results from assumptions, hypotheses, and planned work. Apply this specifically to **Next research step** and record the evidence from the current run.
- **Next experiment:** Change one major factor at a time and compare with a fixed baseline and split. Apply this specifically to **Next research step** and record the evidence from the current run.
- **Acceptance criteria:** Define the evidence required before marking this area complete. Apply this specifically to **Next research step** and record the evidence from the current run.
- **Maintenance plan:** State what should be rechecked when data, code, dependencies, or workflow changes. Apply this specifically to **Next research step** and record the evidence from the current run.

## 11. Recommended next steps

1. Verify exactly when customer remarks are collected and exclude them if they occur after the intended prediction point.
2. Re-run all model variants on one frozen, documented split.
3. Evaluate directly against the persisted test dataset rather than reconstructing indices.
4. Add automated tests for data cleaning, schema validation, preprocessing, and model inference.
5. Add a dependency specification and verify a clean environment installation.
6. Check the dataset license and privacy requirements before publishing any data-derived examples.
7. Add a model card describing intended use, metrics, limitations, and monitoring requirements.
8. Build the proposed API and frontend only after the modeling task has a defensible evaluation design.

## Documentation maintenance

- Append dated experiment notes when new results are produced.
- Mark old results as superseded rather than silently deleting them.
- Separate measured facts, interpretations, assumptions, and roadmap items.
- Never add invented metrics or claim that a planned feature has been implemented.
- Keep customer-level records and credentials out of this public documentation.

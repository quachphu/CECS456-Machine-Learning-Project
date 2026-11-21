# Predicting Loan Default

A CECS 456 final project that explores a loan credit-risk dataset and compares machine learning models for predicting whether a loan will default.

**Authors:** Phu Quach, Om Patel, and David Ramos  
**Course:** CECS 456 · Fall 2026

## Project overview

The project examines a binary target, `loan_status`, where `1` represents default and `0` represents no default. The notebook covers exploratory analysis, data cleaning, preprocessing, model tuning and comparison, feature ablations, test-set evaluation, permutation importance, and error patterns across age and home-ownership groups.

The original dataset contains 32,581 rows and 11 input features. After removing duplicate records and rows with implausible ages or employment lengths, 32,409 rows remain. The cleaned data has a default rate of about 21.9%.

## Repository contents

- `CECS456_Loan_Default_Project.ipynb` — analysis, model training, evaluation, figures, and discussion.
- `credit_risk_dataset 2.csv` — input dataset used by the notebook.

The provided notebook currently loads the CSV with `pd.read_csv("credit_risk_dataset 2.csv")`. Keep that filename in the notebook’s working directory, or update the path in the data-loading cell to match your filename.

## Dataset

The dataset was provided in the **Loan Approval Prediction Dataset-2** ZIP on Canvas. It includes applicant characteristics, loan details, prior credit history, and the target `loan_status`.

Input columns include:

- Applicant: `person_age`, `person_income`, `person_home_ownership`, `person_emp_length`
- Loan: `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`
- Credit history: `cb_person_default_on_file`, `cb_person_cred_hist_length`

The notebook notes missing values in employment length (895 records) and interest rate (3,116 records). These are imputed within the model pipelines using training-fold medians for numeric data and the most frequent value for categorical data. Numeric values are standardized, categorical values are one-hot encoded, and missing numeric values receive indicator features. Income is log-transformed because it is strongly right-skewed.

## Methods

The notebook uses an 80/20 stratified train/test split with `random_state=42`. Models are compared with five-fold stratified cross-validation. Hyperparameters are selected using randomized search scored by average precision (AP), also called PR-AUC in the notebook. Since defaults are the minority class, AP is emphasized over accuracy. Tuning for k-nearest neighbors and RBF SVM uses a random 30% sample of the training set due to runtime; the selected settings are then evaluated on the full training data.

The evaluated approaches are:

- Majority-class baseline
- Logistic Regression
- Decision Tree
- Random Forest
- Histogram-based Gradient Boosting
- k-Nearest Neighbors
- RBF Support Vector Machine

## Results

Gradient Boosting had the highest average precision among the evaluated models. On the held-out test set, it achieved **0.909 AP**, **0.950 ROC-AUC**, **93.97% accuracy**, **98.13% precision**, and **73.84% recall** at the model’s default classification threshold.

| Model | Test AP | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.909 | 0.950 | 0.940 | 0.981 | 0.738 | 0.843 |
| Random Forest | 0.885 | 0.932 | 0.913 | 0.816 | 0.775 | 0.795 |
| Decision Tree | 0.866 | 0.920 | 0.890 | 0.733 | 0.779 | 0.755 |
| RBF SVM | 0.849 | 0.915 | 0.882 | 0.710 | 0.777 | 0.742 |
| k-NN | 0.816 | 0.900 | 0.893 | 0.900 | 0.574 | 0.701 |
| Logistic Regression | 0.712 | 0.876 | 0.807 | 0.540 | 0.795 | 0.643 |
| Majority baseline | 0.219 | 0.500 | 0.781 | 0.000 | 0.000 | 0.000 |

For Gradient Boosting, the high precision means few flagged loans were non-defaults, while recall of 73.8% means the model missed roughly one in four defaults at the default threshold. The notebook also reports that removing both `loan_grade` and `loan_int_rate` reduced test AP to 0.821 in its ablation experiment.

## Run the notebook

1. Clone or download this repository.
2. Install Python 3 and the packages below.
3. Place the CSV next to the notebook and ensure its name matches the `pd.read_csv(...)` path.
4. Open the notebook in Jupyter and run the cells from top to bottom.

```bash
python -m pip install numpy pandas matplotlib scipy scikit-learn notebook ipykernel
jupyter notebook
```

Model tuning can take some time, especially for the ensemble models and cross-validation searches.

## Limitations and responsible use

This is an educational analysis, not a lending product or financial recommendation. The dataset may contain historical patterns and biases, and the model’s predictions should not be used on their own to make real lending decisions. The group error checks by age and renter status are exploratory and do not constitute a complete fairness assessment.

The notebook has already examined its test results, so the test set is no longer a fully untouched benchmark for future model changes. Its cross-validation scores may also be optimistic because hyperparameter selection and model comparison use overlapping training data. Threshold tuning and additional feature ablations are described as future work in the notebook.

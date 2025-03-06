Rubing Zhang: # Predicting Transplant Survival Rates for HCT Patients

## Overview
This competition aims to improve the prediction of transplant survival rates for patients undergoing **allogeneic Hematopoietic Cell Transplantation (HCT)**. The goal is to ensure fair and accurate survival predictions across diverse patient groups, addressing disparities related to **socioeconomic status, race, and geography**.

By using **synthetic data** that mirrors real-world scenarios while protecting patient privacy, participants develop machine learning models to predict survival rates with both **accuracy and fairness**.

---

## 🏆 Competition Objective
Participants are challenged to develop advanced predictive models for **HCT survival outcomes**, with an emphasis on:
- **Enhancing prediction accuracy** using survival analysis techniques.
- **Reducing disparities** in prediction performance across different racial groups.
- **Optimizing model fairness** to ensure equitable healthcare outcomes.

The competition evaluates models using a **Stratified Concordance Index (C-index)**, which measures predictive performance while ensuring equitable predictions across diverse racial groups.

---

## 📌 Implementation Details
This repository contains a Python script (`main.ipynb`) that:
- Processes and preps **HCT survival data**.
- Implements a **Cox Proportional Hazards Model** for survival prediction.
- Evaluates fairness using a **Stratified C-index**.

### **Files in this repository:**
- `main.ipynb`: The primary script for data processing, model training, and evaluation.
- `train.csv`: Training dataset containing patient features and survival outcomes.
- `test.csv`: Test dataset for model evaluation.
- `submission.csv`: Output file with predicted survival probabilities.

### **Key Methods Implemented**
#### **1️⃣ Stratified Concordance Index Calculation**
The **C-index** is computed separately for each racial group, ensuring fairness:

```python
def stratified_concordance_index(y_true, y_pred, race):
    """
    Calculates the stratified concordance index.
    
    Args:
        y_true (pd.Series): True survival times.
        y_pred (pd.Series): Predicted risk scores.
        race (pd.Series): Race of each patient.
    
    Returns:
        float: Stratified concordance index.
    """
    races = race.unique()
    c_indices = []
    for r in races:
        mask = race == r
        c_index = concordance_index(y_true[mask], y_pred[mask])
        c_indices.append(c_index)
    return np.mean(c_indices) - np.std(c_indices)
```

#### **2️⃣ Model Training with Cox Proportional Hazards**
A **Cox Proportional Hazards model** is trained using standard survival analysis techniques:

```python
def train_model(train_df):
    """
    Trains a survival model.
    
    Args:
        train_df (pd.DataFrame): Training data.
    
    Returns:
        model: Trained model.
    """
    # Encode categorical variables
    label_encoder = LabelEncoder()
    train_df['race_encoded'] = label_encoder.fit_transform(train_df['race_group'])

    # Standardize numerical features
    numerical_features = ['age_at_hct', 'donor_age', 'comorbidity_score', 'karnofsky_score']
    scaler = StandardScaler()
    train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])

    # Train Cox model
    cph = CoxPHFitter()
    cph.fit(train_df, duration_col='efs_time', event_col='efs')

    return cph
```

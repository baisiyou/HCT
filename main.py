import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier

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
        try:
            c_index = roc_auc_score(y_true[mask], y_pred[mask])
        except ValueError:
            c_index = np.nan
        c_indices.append(c_index)
    c_indices = [c for c in c_indices if not np.isnan(c)]
    return np.mean(c_indices) - np.std(c_indices) if c_indices else np.nan

def train_model(train_df):
    """
    Trains an XGBoost Classifier model.

    Args:
        train_df (pd.DataFrame): Training data.

    Returns:
        model: Trained model.
    """
    # Preprocess categorical features
    categorical_features = ['race_group']
    label_encoders = {}
    for feature in categorical_features:
        label_encoder = LabelEncoder()
        train_df.loc[:, feature + '_encoded'] = label_encoder.fit_transform(train_df[feature])
        label_encoders[feature] = label_encoder

    # Preprocess numerical features
    numerical_features = ['age_at_hct', 'donor_age', 'comorbidity_score', 'karnofsky_score']
    
    # Replace non-numeric values with NaN
    for feature in numerical_features:
        train_df.loc[:, feature] = pd.to_numeric(train_df[feature], errors='coerce')
    
    # Impute missing values using KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    train_df.loc[:, numerical_features] = imputer.fit_transform(train_df[numerical_features])
    
    # Feature Interactions
    train_df.loc[:, 'age_comorbidity'] = train_df['age_at_hct'] * train_df['comorbidity_score']
    numerical_features.append('age_comorbidity')
        
    scaler = StandardScaler()
    train_df.loc[:, numerical_features] = scaler.fit_transform(train_df[numerical_features])

    # Define features and target
    features = [feature + '_encoded' for feature in categorical_features] + numerical_features
    
    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1]
    }
    
    grid_search = GridSearchCV(XGBClassifier(random_state=42, eval_metric='logloss'), param_grid, cv=3, scoring='roc_auc')
    grid_search.fit(train_df[features], train_df['efs'])
    model = grid_search.best_estimator_

    return model, label_encoders, features, scaler

def predict_risk(model, test_df, label_encoders, features, scaler):
    """
    Predicts risk scores for the test data.

    Args:
        model: Trained model.
        test_df (pd.DataFrame): Test data.
        label_encoders: Label encoders for categorical features.
        features: List of features used in training.
        scaler: Scaler for numerical features.

    Returns:
        pd.Series: Predicted risk scores.
    """
    # Preprocess categorical features
    categorical_features = ['race_group']
    for feature in categorical_features:
        test_df.loc[:, feature + '_encoded'] = label_encoders[feature].transform(test_df[feature])

    # Preprocess numerical features
    numerical_features = ['age_at_hct', 'donor_age', 'comorbidity_score', 'karnofsky_score', 'age_comorbidity']
    # Replace non-numeric values with NaN
    for feature in numerical_features:
        test_df.loc[:, feature] = pd.to_numeric(test_df[feature], errors='coerce')
    # Fill NaN values with the median, handling empty slices
    for feature in numerical_features:
        median_val = test_df[feature].median()
        if not pd.isna(median_val):
            test_df.loc[:, feature] = test_df[feature].fillna(median_val)
        else:
            test_df.loc[:, feature] = test_df[feature].fillna(0) # Fill with 0 if median is NaN
    test_df.loc[:, numerical_features] = scaler.transform(test_df[numerical_features])

    risk_scores = model.predict_proba(test_df[features])[:, 1]
    risk_scores = pd.Series(risk_scores, index=test_df['ID'])
    return risk_scores

def evaluate_model(model, test_df, label_encoders, features, scaler):
    """
    Evaluates the model using stratified concordance index.

    Args:
        model: Trained model.
        test_df (pd.DataFrame): Test data.
        label_encoders: Label encoders for categorical features.
        features: List of features used in training.
        scaler: Scaler for numerical features.

    Returns:
        float: Stratified concordance index.
    """
    risk_scores = predict_risk(model, test_df, label_encoders, features, scaler)
    stratified_c_index = stratified_concordance_index(test_df['efs'], risk_scores, test_df['race_group'])
    return stratified_c_index

def main():
    # Load data
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    # data_dictionary_df = pd.read_csv("data_dictionary.csv") # not used in this version
    # sample_submission_df = pd.read_csv("sample_submission.csv") # not used in this version

    # Train model
    model, label_encoders, features, scaler = train_model(train_df)

    # Evaluate model with Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    c_indices = []
    for train_index, val_index in skf.split(train_df[features], train_df['efs']):
        X_train, X_val = train_df.iloc[train_index], train_df.iloc[val_index]
        y_train, y_val = train_df['efs'].iloc[train_index], train_df['efs'].iloc[val_index]
        model, label_encoders, features, scaler = train_model(X_train)
        risk_scores = predict_risk(model, X_val, label_encoders, features, scaler)
        c_index = stratified_concordance_index(y_val, risk_scores, X_val['race_group'])
        c_indices.append(c_index)
    print(f"Mean Stratified C-index: {np.mean(c_indices)}")

    # Predict risk scores
    risk_scores = predict_risk(model, test_df, label_encoders, features, scaler)

    # Create submission file
    submission_df = pd.DataFrame({'ID': risk_scores.index, 'prediction': risk_scores.values})
    submission_df.to_csv('submission.csv', index=False)
    print("Submission file created: submission.csv")

if __name__ == "__main__":
    main()
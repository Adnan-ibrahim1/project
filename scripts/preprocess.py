from scripts.database import get_data
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif


def load_data():
    query = "SELECT * FROM dataset"
    data = get_data(query)
    return data

def handle_missing_data(data):
    """
    Fill missing data with zero (you can change this as per need).
    """
    return data.fillna(0)

def encode_target(data):
    """
    Convert target column to categorical.
    """
    data['target'] = data['target'].astype('category')
    return data

def preprocess_data(X, y):
    """
    This function takes the feature data (X) and target data (y),
    imputes missing values, and returns processed data.
    """

    # Handle missing values (Imputation)
    imputer = SimpleImputer(strategy='mean')  # You can change strategy based on your needs
    X_imputed = imputer.fit_transform(X)

    # Feature selection
    selector = SelectKBest(score_func=f_classif, k='all')  # Adjust 'k' based on how many features you want to keep
    X_selected = selector.fit_transform(X_imputed, y)

    return X_selected, y

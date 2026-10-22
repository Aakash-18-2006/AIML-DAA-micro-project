"""
Data Preprocessing Module for Smart Study Path
Handles dataset loading, validation, feature extraction, and train-test splitting.
"""

import os
import pandas as pd
import numpy as np

FEATURE_COLUMNS = [
    "previous_score",
    "practice_hours",
    "questions_solved",
    "previous_attempts",
]
TARGET_COLUMN = "target_score"


def load_dataset(file_path: str = "data/student_data.csv") -> pd.DataFrame:
    """
    Load student performance dataset from a CSV file.
    Validates existence, column structure, and cleans any missing records.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    df = pd.read_csv(file_path)

    # Check required columns
    required_cols = ["topic"] + FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Drop missing values if any
    df = df.dropna()

    # Validate numeric types
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    return df


def prepare_features(df: pd.DataFrame):
    """
    Extract feature matrix X and target vector y from dataframe.
    """
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=float)
    return X, y


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """
    Splits arrays into random train and test subsets.
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(X)
    n_test = int(np.round(n_samples * test_size))

    shuffled_indices = rng.permutation(n_samples)
    test_indices = shuffled_indices[:n_test]
    train_indices = shuffled_indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def get_train_test_data(file_path: str = "data/student_data.csv", test_size: float = 0.2, random_state: int = 42):
    """
    Loads data and returns train/test splits.
    """
    df = load_dataset(file_path)
    X, y = prepare_features(df)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

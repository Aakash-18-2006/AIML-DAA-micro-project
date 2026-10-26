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


DEFAULT_STUDENT_CSV_DATA = """topic,previous_score,practice_hours,questions_solved,previous_attempts,target_score
Arrays,65,12,35,1,72
Arrays,45,6,15,2,52
Arrays,80,18,50,1,88
Arrays,30,4,10,3,38
Arrays,55,10,25,1,63
Arrays,70,14,40,2,76
Arrays,40,5,12,3,48
Sorting,58,8,22,2,65
Sorting,35,4,10,3,42
Sorting,75,16,45,1,82
Sorting,48,7,18,2,55
Sorting,62,11,30,1,69
Sorting,25,3,8,4,32
Sorting,82,20,55,1,90
Searching,60,9,26,1,68
Searching,42,5,14,2,49
Searching,78,15,44,1,84
Searching,33,4,9,3,39
Searching,52,8,20,2,59
Searching,70,13,38,1,77
Searching,46,6,16,2,53
Graphs,40,6,14,2,47
Graphs,68,14,38,1,75
Graphs,32,4,8,3,38
Graphs,55,10,24,2,62
Graphs,75,17,48,1,83
Graphs,28,3,6,4,33
Graphs,62,12,30,1,70
BFS,48,8,18,2,55
BFS,35,5,10,3,43
BFS,72,15,40,1,80
BFS,58,11,28,2,66
BFS,40,6,12,3,48
BFS,80,18,52,1,87
BFS,25,3,6,4,31
DFS,50,9,20,2,58
DFS,38,5,11,3,45
DFS,70,14,36,1,78
DFS,60,11,26,2,67
DFS,42,6,14,3,49
DFS,78,17,46,1,85
DFS,30,4,8,4,36
Dijkstra,35,5,10,3,42
Dijkstra,22,3,5,4,28
Dijkstra,65,13,32,2,72
Dijkstra,48,8,18,2,56
Dijkstra,72,16,42,1,81
Dijkstra,30,4,8,3,37
Dijkstra,58,11,25,2,65
Dynamic Programming,55,10,22,2,63
Dynamic Programming,38,5,10,3,44
Dynamic Programming,75,16,42,1,82
Dynamic Programming,48,8,16,2,55
Dynamic Programming,65,13,30,1,73
Dynamic Programming,30,4,8,4,35
Dynamic Programming,82,19,50,1,89
Dynamic Programming,60,11,25,2,68
"""


def resolve_dataset_path(file_path: str = "data/student_data.csv") -> str:
    """
    Resolves dataset file path robustly on both Windows and Linux,
    checking relative to CWD and relative to workspace/repository root.
    """
    if file_path and os.path.exists(file_path):
        return file_path

    # Try relative to repository root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(base_dir, file_path if file_path else "data/student_data.csv")
    if os.path.exists(candidate):
        return candidate

    candidate_data = os.path.join(base_dir, "data", os.path.basename(file_path if file_path else "student_data.csv"))
    if os.path.exists(candidate_data):
        return candidate_data

    return candidate_data


def load_dataset(file_path: str = "data/student_data.csv") -> pd.DataFrame:
    """
    Load student performance dataset from a CSV file.
    Validates existence, column structure, and cleans any missing records.
    If the file is missing, automatically generates the default dataset safely.
    """
    resolved_path = resolve_dataset_path(file_path)

    if not os.path.exists(resolved_path):
        # Auto-create data directory and fallback dataset if missing
        os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
        with open(resolved_path, "w", encoding="utf-8") as f:
            f.write(DEFAULT_STUDENT_CSV_DATA.strip())

    df = pd.read_csv(resolved_path)

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

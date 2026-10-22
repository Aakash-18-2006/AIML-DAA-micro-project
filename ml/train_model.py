"""
Model Training Module for Smart Study Path
Implements Multiple Linear Regression using Ordinary Least Squares (OLS).
Computes performance metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.
"""

from typing import Dict, Tuple, Any
import os
import sys
import numpy as np

# Ensure workspace root is in sys.path when executed directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ml.preprocessing import get_train_test_data


class MultipleLinearRegression:
    """
    Multiple Linear Regression trained via Ordinary Least Squares (OLS).
    Equation: y = X * w + b
    Closed-form solution: w_aug = (X_aug^T * X_aug)^(-1) * X_aug^T * y
    """

    def __init__(self):
        self.coef_: np.ndarray = np.array([])
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultipleLinearRegression":
        """
        Fits the linear model using Ordinary Least Squares closed-form normal equation.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Augment X with leading column of ones for intercept
        n_samples = X.shape[0]
        X_aug = np.hstack([np.ones((n_samples, 1)), X])

        # Compute optimal weights using pseudo-inverse: (X^T * X)^(-1) * X^T * y
        w_aug = np.linalg.pinv(X_aug) @ y

        self.intercept_ = float(w_aug[0])
        self.coef_ = np.asarray(w_aug[1:], dtype=float)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts continuous target values given feature matrix X.
        """
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot != 0 else 1.0

    return mae, mse, r2


def train_linear_regression(
    file_path: str = "data/student_data.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[MultipleLinearRegression, Dict[str, Any]]:
    """
    Train a Multiple Linear Regression model on student academic metrics.

    Returns:
        model: Trained MultipleLinearRegression model
        metrics: Dictionary containing MAE, MSE, R2 score, sample counts,
                 coefficients, and intercept.
    """
    X_train, X_test, y_train, y_test = get_train_test_data(
        file_path=file_path, test_size=test_size, random_state=random_state
    )

    model = MultipleLinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae, mse, r2 = compute_metrics(y_test, y_pred)

    metrics = {
        "mae": round(mae, 2),
        "mse": round(mse, 2),
        "r2_score": round(r2, 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "coefficients": [round(float(c), 4) for c in model.coef_],
        "intercept": round(float(model.intercept_), 2),
    }

    return model, metrics


if __name__ == "__main__":
    trained_model, eval_metrics = train_linear_regression()
    print("--- Model Training Successful ---")
    print(f"MAE: {eval_metrics['mae']}")
    print(f"MSE: {eval_metrics['mse']}")
    print(f"R² Score: {eval_metrics['r2_score']}")
    print(f"Coefficients: {eval_metrics['coefficients']}")
    print(f"Intercept: {eval_metrics['intercept']}")

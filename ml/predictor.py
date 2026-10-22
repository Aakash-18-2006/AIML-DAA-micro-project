"""
Prediction Module for Smart Study Path
Takes student input metrics, uses the trained ML model to predict scores,
and flags weak topics based on the performance threshold.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

DEFAULT_WEAK_THRESHOLD = 60.0


def validate_student_metric(
    previous_score: float,
    practice_hours: float,
    questions_solved: int,
    previous_attempts: int,
) -> None:
    """
    Validates input ranges for student metrics.
    Raises ValueError on invalid bounds.
    """
    if not (0 <= previous_score <= 100):
        raise ValueError(f"Previous score must be between 0 and 100 (got {previous_score}).")
    if practice_hours < 0:
        raise ValueError(f"Practice hours cannot be negative (got {practice_hours}).")
    if questions_solved < 0:
        raise ValueError(f"Questions solved cannot be negative (got {questions_solved}).")
    if previous_attempts < 0:
        raise ValueError(f"Previous attempts cannot be negative (got {previous_attempts}).")


def predict_topic_score(
    model: Any,
    previous_score: float,
    practice_hours: float,
    questions_solved: int,
    previous_attempts: int,
) -> float:
    """
    Predicts the expected performance score for a single topic using Multiple Linear Regression.
    Clamps prediction safely between 0.0 and 100.0.
    """
    validate_student_metric(previous_score, practice_hours, questions_solved, previous_attempts)

    # Features: [previous_score, practice_hours, questions_solved, previous_attempts]
    feature_vector = np.array([[previous_score, practice_hours, questions_solved, previous_attempts]], dtype=float)
    raw_pred = float(model.predict(feature_vector)[0])

    # Ensure realistic academic score bounds [0.0, 100.0]
    clamped_score = max(0.0, min(100.0, raw_pred))
    return round(clamped_score, 1)


def is_weak_topic(score: float, threshold: float = DEFAULT_WEAK_THRESHOLD) -> bool:
    """
    Returns True if score is strictly below the weak performance threshold (default 60%).
    """
    return score < threshold


def predict_all_topics(
    model: Any,
    student_inputs: Dict[str, Dict[str, Any]],
    threshold: float = DEFAULT_WEAK_THRESHOLD,
) -> Dict[str, Dict[str, Any]]:
    """
    Computes predicted scores and classification for all topics provided by the student.

    Args:
        model: Trained LinearRegression model
        student_inputs: Dictionary mapping topic_name -> {
            'previous_score': float,
            'practice_hours': float,
            'questions_solved': int,
            'previous_attempts': int
        }
        threshold: Score threshold for weakness

    Returns:
        Dictionary mapping topic_name -> {
            'predicted_score': float,
            'status': 'Weak' | 'Good',
            'is_weak': bool,
            'inputs': dict
        }
    """
    predictions = {}
    for topic, metrics in student_inputs.items():
        score = predict_topic_score(
            model=model,
            previous_score=float(metrics["previous_score"]),
            practice_hours=float(metrics["practice_hours"]),
            questions_solved=int(metrics["questions_solved"]),
            previous_attempts=int(metrics["previous_attempts"]),
        )
        weak = is_weak_topic(score, threshold=threshold)
        predictions[topic] = {
            "predicted_score": score,
            "status": "Weak" if weak else "Good",
            "is_weak": weak,
            "inputs": metrics,
        }
    return predictions

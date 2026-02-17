"""
Calculo de metricas de evaluacion para todas las tareas de RE.
"""

from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report
)
import numpy as np
from typing import List, Dict


def calculate_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """
    Calcula metricas de clasificacion binaria (F/NF).

    Args:
        y_true: Etiquetas reales (F/NF)
        y_pred: Etiquetas predichas (F/NF)

    Returns:
        Diccionario con precision, recall, f1, accuracy
    """
    valid_indices = [i for i, p in enumerate(y_pred) if p in ("F", "NF")]

    if len(valid_indices) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "valid_predictions": 0,
            "total_predictions": len(y_pred),
            "invalid_rate": 1.0
        }

    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]

    # Convertir a binario (F=1, NF=0)
    y_true_bin = [1 if y == "F" else 0 for y in y_true_valid]
    y_pred_bin = [1 if y == "F" else 0 for y in y_pred_valid]

    return {
        "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "valid_predictions": len(valid_indices),
        "total_predictions": len(y_pred),
        "invalid_rate": 1 - (len(valid_indices) / len(y_pred))
    }


def calculate_binary_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict[str, float]:
    """
    Calcula metricas para tareas binarias (ambiguity, completeness, testability, inconsistency).

    Args:
        y_true: Etiquetas reales (True/False)
        y_pred: Etiquetas predichas (True/False)

    Returns:
        Diccionario con precision, recall, f1, accuracy
    """
    if not y_true:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0,
                "valid_predictions": 0, "total_predictions": 0, "invalid_rate": 0.0}

    y_true_int = [int(y) for y in y_true]
    y_pred_int = [int(y) for y in y_pred]

    return {
        "precision": precision_score(y_true_int, y_pred_int, zero_division=0),
        "recall": recall_score(y_true_int, y_pred_int, zero_division=0),
        "f1": f1_score(y_true_int, y_pred_int, zero_division=0),
        "accuracy": accuracy_score(y_true_int, y_pred_int),
        "valid_predictions": len(y_true),
        "total_predictions": len(y_true),
        "invalid_rate": 0.0
    }


def calculate_aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Calcula media y desviacion estandar de metricas de multiples iteraciones.

    Args:
        metrics_list: Lista de diccionarios de metricas

    Returns:
        Diccionario con media y std de cada metrica
    """
    if not metrics_list:
        return {}

    result = {}
    metric_names = ["precision", "recall", "f1", "accuracy"]

    for metric in metric_names:
        values = [m[metric] for m in metrics_list if metric in m]
        if values:
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_std"] = float(np.std(values))

    # Aggregate invalid rate if present
    invalid_rates = [m.get("invalid_rate", 0) for m in metrics_list]
    result["invalid_rate_mean"] = float(np.mean(invalid_rates))

    return result


def get_confusion_matrix(y_true: List[str], y_pred: List[str]) -> np.ndarray:
    """Genera matriz de confusion para clasificacion F/NF."""
    valid_indices = [i for i, p in enumerate(y_pred) if p in ("F", "NF")]
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]

    y_true_bin = [1 if y == "F" else 0 for y in y_true_valid]
    y_pred_bin = [1 if y == "F" else 0 for y in y_pred_valid]

    return confusion_matrix(y_true_bin, y_pred_bin)


def get_classification_report_text(y_true: List[str], y_pred: List[str]) -> str:
    """Genera un reporte de clasificacion completo en texto."""
    valid_indices = [i for i, p in enumerate(y_pred) if p in ("F", "NF")]
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]

    return classification_report(y_true_valid, y_pred_valid, zero_division=0)

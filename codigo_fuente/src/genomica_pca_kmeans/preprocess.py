"""Preprocesamiento de la matriz de expresion."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Resultado de seleccionar genes por varianza."""

    X_selected: pd.DataFrame
    selected_variances: pd.Series


@dataclass(frozen=True)
class ScalingResult:
    """Resultado de estandarizar variables."""

    X_scaled: pd.DataFrame
    scaler: StandardScaler


def select_high_variance_genes(
    X: pd.DataFrame,
    *,
    n_top_genes: int | None = 5000,
) -> FeatureSelectionResult:
    """Selecciona genes con mayor varianza antes de estandarizar."""

    if n_top_genes is not None and n_top_genes <= 0:
        raise ValueError("n_top_genes debe ser positivo o None.")

    variances = X.var(axis=0, ddof=0).sort_values(ascending=False)
    if n_top_genes is None or n_top_genes >= X.shape[1]:
        selected_genes = variances.index
    else:
        selected_genes = variances.head(n_top_genes).index

    return FeatureSelectionResult(
        X_selected=X.loc[:, selected_genes].copy(),
        selected_variances=variances.loc[selected_genes],
    )


def standardize_features(X: pd.DataFrame) -> ScalingResult:
    """Estandariza cada gen con media 0 y desviacion estandar 1."""

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(scaled_values, index=X.index, columns=X.columns)
    return ScalingResult(X_scaled=X_scaled, scaler=scaler)

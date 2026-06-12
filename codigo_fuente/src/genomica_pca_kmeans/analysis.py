"""PCA, K-Means y comparacion de clusters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_score


@dataclass(frozen=True)
class PCAResult:
    """Resultado de PCA."""

    scores: pd.DataFrame
    model: PCA
    explained_variance_ratio: pd.Series
    cumulative_variance: pd.Series


@dataclass(frozen=True)
class KMeansResult:
    """Resultado de evaluar y ajustar K-Means."""

    evaluation: pd.DataFrame
    best_k: int
    labels: pd.Series
    model: KMeans


def run_pca(
    X_scaled: pd.DataFrame,
    *,
    n_components: int = 10,
    random_state: int = 42,
) -> PCAResult:
    """Ajusta PCA sobre la matriz estandarizada."""

    max_components = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
    if max_components < 2:
        raise ValueError("PCA requiere al menos dos componentes posibles.")

    model = PCA(n_components=max_components, random_state=random_state)
    scores_array = model.fit_transform(X_scaled)
    columns = [f"PC{index}" for index in range(1, max_components + 1)]
    scores = pd.DataFrame(scores_array, index=X_scaled.index, columns=columns)
    explained = pd.Series(model.explained_variance_ratio_, index=columns, name="explained_variance")
    cumulative = explained.cumsum()
    cumulative.name = "cumulative_variance"
    return PCAResult(
        scores=scores,
        model=model,
        explained_variance_ratio=explained,
        cumulative_variance=cumulative,
    )


def evaluate_kmeans(
    embedding: pd.DataFrame,
    *,
    k_values: Iterable[int] = range(2, 11),
    random_state: int = 42,
    n_init: int = 20,
) -> KMeansResult:
    """Evalua K-Means con metricas internas y ajusta el mejor k por Davies-Bouldin."""

    valid_k_values = [int(k) for k in k_values if 2 <= int(k) < len(embedding)]
    if not valid_k_values:
        raise ValueError("Se requiere al menos un k valido entre 2 y n_muestras - 1.")

    rows: list[dict[str, float | int]] = []
    matrix = embedding.to_numpy()
    fitted_models: dict[int, KMeans] = {}
    for k in valid_k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(matrix)
        db_score = davies_bouldin_score(matrix, labels)
        sil_score = silhouette_score(matrix, labels)
        rows.append({"k": k, "davies_bouldin": float(db_score), "silhouette": float(sil_score)})
        fitted_models[k] = model

    evaluation = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    best_row = evaluation.loc[evaluation["davies_bouldin"].idxmin()]
    best_k = int(best_row["k"])
    best_model = fitted_models[best_k]
    labels = pd.Series(best_model.labels_, index=embedding.index, name="cluster")
    return KMeansResult(evaluation=evaluation, best_k=best_k, labels=labels, model=best_model)


def build_contingency(y_true: pd.Series, clusters: pd.Series) -> pd.DataFrame:
    """Construye tabla de contingencia tipo tumoral contra cluster."""

    y_true = pd.Series(y_true, name="tipo_tumor").astype(str)
    clusters = pd.Series(clusters, name="cluster").astype(str)
    if len(y_true) != len(clusters):
        raise ValueError("y_true y clusters deben tener la misma longitud.")
    y_true.index = clusters.index
    return pd.crosstab(y_true, clusters)


def evaluate_cluster_label_alignment(y_true: pd.Series, clusters: pd.Series) -> dict[str, float]:
    """Calcula metricas externas entre clusters y etiquetas reales."""

    y_true = pd.Series(y_true, name="tipo_tumor").astype(str)
    clusters = pd.Series(clusters, name="cluster").astype(str)
    if len(y_true) != len(clusters):
        raise ValueError("y_true y clusters deben tener la misma longitud.")
    y_true.index = clusters.index

    contingency = pd.crosstab(y_true, clusters)
    total = int(contingency.to_numpy().sum())
    purity = float(contingency.max(axis=0).sum() / total) if total else 0.0

    return {
        "purity": purity,
        "ari": float(adjusted_rand_score(y_true, clusters)),
        "nmi": float(normalized_mutual_info_score(y_true, clusters)),
    }

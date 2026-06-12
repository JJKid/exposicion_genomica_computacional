"""Generacion de figuras del analisis exploratorio."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def save_class_counts(y: pd.Series, output_path: str | Path) -> Path:
    counts = y.astype(str).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(counts.index, counts.values, color="#4C78A8")
    ax.set_xlabel("Tipo tumoral")
    ax.set_ylabel("Numero de muestras")
    ax.set_title("Conteo de muestras por tipo tumoral")
    for index, value in enumerate(counts.values):
        ax.text(index, value, str(value), ha="center", va="bottom", fontsize=9)
    return _save(fig, output_path)


def save_explained_variance(explained: pd.Series, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(1, len(explained) + 1)
    ax.bar(x, explained.values, color="#59A14F")
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Varianza explicada")
    ax.set_title("Varianza explicada por componente")
    ax.set_xticks(list(x))
    ax.set_xticklabels(explained.index)
    return _save(fig, output_path)


def save_cumulative_variance(cumulative: pd.Series, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(1, len(cumulative) + 1)
    ax.plot(x, cumulative.values, marker="o", color="#E15759")
    ax.set_xlabel("Numero de componentes")
    ax.set_ylabel("Varianza acumulada")
    ax.set_title("Varianza explicada acumulada")
    ax.set_xticks(list(x))
    ax.set_ylim(0, min(1.05, max(0.05, cumulative.max() + 0.1)))
    ax.grid(alpha=0.25)
    return _save(fig, output_path)


def save_pca_by_label(scores: pd.DataFrame, labels: pd.Series, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _scatter_by_category(ax, scores, labels.astype(str))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA coloreado por tipo tumoral")
    ax.legend(title="Tipo", bbox_to_anchor=(1.02, 1), loc="upper left")
    return _save(fig, output_path)


def save_davies_bouldin(evaluation: pd.DataFrame, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(evaluation["k"], evaluation["davies_bouldin"], marker="o", color="#B07AA1")
    best_row = evaluation.loc[evaluation["davies_bouldin"].idxmin()]
    ax.scatter(best_row["k"], best_row["davies_bouldin"], s=90, color="#F28E2B", zorder=3)
    ax.set_xlabel("Numero de clusters (k)")
    ax.set_ylabel("Indice Davies-Bouldin")
    ax.set_title("Seleccion de k: Davies-Bouldin y Silhouette")
    ax.grid(alpha=0.25)
    if "silhouette" in evaluation:
        ax2 = ax.twinx()
        ax2.plot(evaluation["k"], evaluation["silhouette"], marker="s", color="#1F6F78")
        best_silhouette = evaluation.loc[evaluation["silhouette"].idxmax()]
        ax2.scatter(best_silhouette["k"], best_silhouette["silhouette"], s=70, color="#59A14F", zorder=3)
        ax2.set_ylabel("Silhouette")
        lines = ax.get_lines() + ax2.get_lines()
        labels = ["Davies-Bouldin (menor es mejor)", "Silhouette (mayor es mejor)"]
        ax.legend(lines, labels, loc="best", fontsize=8)
    return _save(fig, output_path)


def save_clusters(scores: pd.DataFrame, clusters: pd.Series, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _scatter_by_category(ax, scores, clusters.astype(str))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA coloreado por cluster K-Means")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    return _save(fig, output_path)


def save_contingency_matrix(contingency: pd.DataFrame, output_path: str | Path) -> Path:
    fig_width = max(6, 1.1 * contingency.shape[1] + 3)
    fig_height = max(4.5, 0.7 * contingency.shape[0] + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(contingency.values, cmap="Blues")
    ax.set_xticks(range(contingency.shape[1]))
    ax.set_xticklabels(contingency.columns)
    ax.set_yticks(range(contingency.shape[0]))
    ax.set_yticklabels(contingency.index)
    ax.set_xlabel("Cluster K-Means")
    ax.set_ylabel("Tipo tumoral real")
    ax.set_title("Matriz de contingencia")
    for row in range(contingency.shape[0]):
        for col in range(contingency.shape[1]):
            value = int(contingency.iat[row, col])
            ax.text(col, row, str(value), ha="center", va="center", color="#111111", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, output_path)


def _scatter_by_category(ax: plt.Axes, scores: pd.DataFrame, categories: pd.Series) -> None:
    if "PC1" not in scores or "PC2" not in scores:
        raise ValueError("scores debe contener columnas PC1 y PC2.")
    categories = pd.Series(categories.to_numpy(), index=scores.index, name=categories.name)
    for category in sorted(categories.unique()):
        mask = categories == category
        ax.scatter(
            scores.loc[mask, "PC1"],
            scores.loc[mask, "PC2"],
            s=36,
            alpha=0.82,
            label=category,
            edgecolors="white",
            linewidths=0.35,
        )


def _save(fig: plt.Figure, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path

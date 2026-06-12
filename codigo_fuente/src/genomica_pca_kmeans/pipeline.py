"""Pipeline ejecutable de RNA-Seq, PCA y K-Means."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import pandas as pd

from .analysis import (
    KMeansResult,
    PCAResult,
    build_contingency,
    evaluate_cluster_label_alignment,
    evaluate_kmeans,
    run_pca,
)
from .data import DatasetError, ExpressionDataset, load_expression_data, validate_expression_dataset
from .plots import (
    save_class_counts,
    save_clusters,
    save_contingency_matrix,
    save_cumulative_variance,
    save_davies_bouldin,
    save_explained_variance,
    save_pca_by_label,
)
from .preprocess import FeatureSelectionResult, ScalingResult, select_high_variance_genes, standardize_features


@dataclass(frozen=True)
class PipelineResult:
    """Objetos principales generados por el pipeline."""

    dataset: ExpressionDataset
    summary: dict[str, object]
    feature_selection: FeatureSelectionResult
    scaling: ScalingResult
    pca: PCAResult
    kmeans: KMeansResult
    contingency: pd.DataFrame
    figure_paths: dict[str, Path]


def run_pipeline(
    *,
    data_dir: str | Path = "data",
    figures_dir: str | Path = "figures",
    data_csv: str | Path | None = None,
    labels_csv: str | Path | None = None,
    download: bool = True,
    results_dir: str | Path = "results",
    n_top_genes: int | None = 5000,
    n_components: int = 10,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
) -> PipelineResult:
    """Ejecuta el flujo completo y guarda las figuras solicitadas."""

    dataset = load_expression_data(
        data_dir=data_dir,
        data_csv=data_csv,
        labels_csv=labels_csv,
        download=download,
    )
    summary = validate_expression_dataset(dataset)

    figures_dir = Path(figures_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: dict[str, Path] = {}
    figure_paths["conteo_clases"] = save_class_counts(dataset.y, figures_dir / "conteo_clases.png")

    feature_selection = select_high_variance_genes(dataset.X, n_top_genes=n_top_genes)
    scaling = standardize_features(feature_selection.X_selected)
    pca = run_pca(scaling.X_scaled, n_components=n_components, random_state=random_state)

    figure_paths["varianza_pca"] = save_explained_variance(pca.explained_variance_ratio, figures_dir / "varianza_pca.png")
    figure_paths["varianza_acumulada"] = save_cumulative_variance(
        pca.cumulative_variance,
        figures_dir / "varianza_acumulada.png",
    )
    figure_paths["pca_tipos_tumor"] = save_pca_by_label(pca.scores, dataset.y, figures_dir / "pca_tipos_tumor.png")

    kmeans = evaluate_kmeans(
        pca.scores,
        k_values=range(k_min, k_max + 1),
        random_state=random_state,
    )
    figure_paths["davies_bouldin"] = save_davies_bouldin(kmeans.evaluation, figures_dir / "davies_bouldin.png")
    figure_paths["kmeans_clusters"] = save_clusters(pca.scores, kmeans.labels, figures_dir / "kmeans_clusters.png")

    contingency = build_contingency(dataset.y, kmeans.labels)
    figure_paths["matriz_contingencia"] = save_contingency_matrix(
        contingency,
        figures_dir / "matriz_contingencia.png",
    )
    label_metrics = evaluate_cluster_label_alignment(dataset.y, kmeans.labels)

    summary = {
        **summary,
        "n_selected_genes": int(feature_selection.X_selected.shape[1]),
        "n_pca_components": int(pca.scores.shape[1]),
        "best_k": int(kmeans.best_k),
        "best_davies_bouldin": float(kmeans.evaluation["davies_bouldin"].min()),
        "best_silhouette": float(
            kmeans.evaluation.loc[kmeans.evaluation["silhouette"].idxmax(), "silhouette"]
        ),
        "silhouette_at_best_k": float(
            kmeans.evaluation.loc[kmeans.evaluation["k"] == kmeans.best_k, "silhouette"].iloc[0]
        ),
        "cluster_label_metrics": label_metrics,
        "pca_explained_variance": {
            str(component): float(value) for component, value in pca.explained_variance_ratio.items()
        },
        "pca_cumulative_variance": {
            str(component): float(value) for component, value in pca.cumulative_variance.items()
        },
    }
    _save_result_tables(
        summary=summary,
        kmeans_evaluation=kmeans.evaluation,
        contingency=contingency,
        label_metrics=label_metrics,
        pca_explained=pca.explained_variance_ratio,
        pca_cumulative=pca.cumulative_variance,
        output_dir=results_dir,
    )
    return PipelineResult(
        dataset=dataset,
        summary=summary,
        feature_selection=feature_selection,
        scaling=scaling,
        pca=pca,
        kmeans=kmeans,
        contingency=contingency,
        figure_paths=figure_paths,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline RNA-Seq: varianza, PCA y K-Means.")
    parser.add_argument("--data-dir", default="data", help="Directorio donde se buscara o descargara el dataset.")
    parser.add_argument("--figures-dir", default="figures", help="Directorio de salida para figuras PNG.")
    parser.add_argument("--results-dir", default="results", help="Directorio de salida para tablas y resumen.")
    parser.add_argument("--data-csv", default=None, help="Ruta local a data.csv.")
    parser.add_argument("--labels-csv", default=None, help="Ruta local a labels.csv.")
    parser.add_argument("--no-download", action="store_true", help="No intentar descargar datos desde UCI.")
    parser.add_argument("--n-top-genes", type=int, default=5000, help="Numero de genes de mayor varianza.")
    parser.add_argument("--n-components", type=int, default=10, help="Numero maximo de componentes PCA.")
    parser.add_argument("--k-min", type=int, default=2, help="Primer valor de k a evaluar.")
    parser.add_argument("--k-max", type=int, default=10, help="Ultimo valor de k a evaluar.")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla para reproducibilidad.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = run_pipeline(
            data_dir=args.data_dir,
            figures_dir=args.figures_dir,
            data_csv=args.data_csv,
            labels_csv=args.labels_csv,
            download=not args.no_download,
            results_dir=args.results_dir,
            n_top_genes=args.n_top_genes,
            n_components=args.n_components,
            k_min=args.k_min,
            k_max=args.k_max,
            random_state=args.random_state,
        )
    except DatasetError as exc:
        print(f"Error de datos: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Error de parametros: {exc}", file=sys.stderr)
        return 2

    print("Pipeline completado.")
    print(f"Muestras: {result.summary['n_samples']}")
    print(f"Genes originales: {result.summary['n_genes']}")
    print(f"Genes seleccionados: {result.summary['n_selected_genes']}")
    print(f"Mejor k por Davies-Bouldin: {result.summary['best_k']}")
    print(f"Silhouette en mejor k: {result.summary['silhouette_at_best_k']:.4f}")
    print(f"Figuras guardadas en: {Path(args.figures_dir).resolve()}")
    print(f"Resultados guardados en: {Path(args.results_dir).resolve()}")
    return 0


def _save_result_tables(
    *,
    summary: dict[str, object],
    kmeans_evaluation: pd.DataFrame,
    contingency: pd.DataFrame,
    label_metrics: dict[str, float],
    pca_explained: pd.Series,
    pca_cumulative: pd.Series,
    output_dir: Path,
) -> None:
    """Guarda salidas tabulares usadas por el reporte y la presentacion."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    kmeans_evaluation.to_csv(output_dir / "davies_bouldin.csv", index=False)
    kmeans_evaluation.to_csv(output_dir / "cluster_metrics.csv", index=False)
    (output_dir / "cluster_label_metrics.json").write_text(
        json.dumps(label_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    contingency.to_csv(output_dir / "matriz_contingencia.csv")
    pd.DataFrame(
        {
            "component": list(pca_explained.index),
            "explained_variance": pca_explained.to_numpy(),
            "cumulative_variance": pca_cumulative.to_numpy(),
        }
    ).to_csv(output_dir / "pca_variance.csv", index=False)


if __name__ == "__main__":
    raise SystemExit(main())

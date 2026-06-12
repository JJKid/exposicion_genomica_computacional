"""Carga, descarga y validacion del dataset de expresion genica."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tarfile
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve
import zipfile

import pandas as pd
from pandas.api.types import is_numeric_dtype

DEFAULT_UCI_URL = (
    "https://archive.ics.uci.edu/static/public/401/"
    "gene%2Bexpression%2Bcancer%2Brna%2Bseq.zip"
)
LEGACY_UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00401/"
    "TCGA-PANCAN-HiSeq-801x20531.tar.gz"
)

EXPECTED_UCI_SHAPE = (801, 20531)
EXPECTED_UCI_CLASSES = {"BRCA", "COAD", "KIRC", "LUAD", "PRAD"}


class DatasetError(RuntimeError):
    """Error recuperable relacionado con la estructura del dataset."""


class DatasetDownloadError(DatasetError):
    """Error recuperable relacionado con la descarga del dataset."""


@dataclass(frozen=True)
class DatasetFiles:
    """Rutas a los archivos CSV requeridos."""

    data_csv: Path
    labels_csv: Path


@dataclass(frozen=True)
class ExpressionDataset:
    """Matriz de expresion y etiquetas tumorales ya alineadas."""

    X: pd.DataFrame
    y: pd.Series
    sample_ids: pd.Index
    data_csv: Path | None = None
    labels_csv: Path | None = None


def ensure_dataset(
    data_dir: str | Path = "data",
    *,
    download: bool = True,
    urls: Iterable[str] = (DEFAULT_UCI_URL, LEGACY_UCI_URL),
) -> DatasetFiles:
    """Localiza el dataset o lo descarga desde UCI si hace falta.

    La funcion acepta tanto el ZIP actual del repositorio UCI como el TAR.GZ
    historico. Si no hay red, el error explica como usar archivos locales.
    """

    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    existing = find_dataset_files(raw_dir)
    if existing is not None:
        return existing

    if not download:
        raise DatasetError(
            "No se encontraron data.csv y labels.csv en "
            f"{raw_dir}. Descarga el dataset manualmente o pasa "
            "--data-csv y --labels-csv junto con --no-download."
        )

    raw_dir.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for index, url in enumerate(urls, start=1):
        suffix = ".tar.gz" if url.endswith(".tar.gz") else ".zip"
        archive_path = raw_dir / f"uci_gene_expression_{index}{suffix}"
        try:
            download_file(url, archive_path)
            extract_archive_tree(archive_path, raw_dir)
            found = find_dataset_files(raw_dir)
            if found is not None:
                return found
        except DatasetDownloadError as exc:
            last_error = exc

    raise DatasetDownloadError(
        "No se pudo descargar o extraer el dataset de UCI. "
        "Si no hay conexion, descarga manualmente el archivo desde "
        "https://archive.ics.uci.edu/dataset/401/gene%2Bexpression%2Bcancer%2Brna%2Bseq "
        "y coloca data.csv y labels.csv bajo data/raw/, o usa "
        "--data-csv y --labels-csv. "
        f"Ultimo error: {last_error}"
    )


def download_file(url: str, output_path: str | Path) -> Path:
    """Descarga un archivo si no existe localmente."""

    output_path = Path(output_path)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(url, output_path)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise DatasetDownloadError(f"No se pudo descargar {url}: {exc}") from exc
    return output_path


def extract_archive_tree(archive_path: str | Path, destination: str | Path) -> None:
    """Extrae un archivo y cualquier archivo comprimido anidado reconocido."""

    archive_path = Path(archive_path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    queue = [archive_path]
    processed: set[Path] = set()
    while queue:
        current = queue.pop(0)
        resolved = current.resolve()
        if resolved in processed or not current.exists():
            continue
        processed.add(resolved)

        extracted = _extract_one_archive(current, destination if current == archive_path else current.parent)
        queue.extend(path for path in extracted if _is_supported_archive(path))


def find_dataset_files(search_dir: str | Path) -> DatasetFiles | None:
    """Busca data.csv y labels.csv dentro de un directorio."""

    search_dir = Path(search_dir)
    if not search_dir.exists():
        return None

    csv_files = [path for path in search_dir.rglob("*.csv") if path.is_file()]
    data_csv = _first_named(csv_files, {"data.csv"})
    labels_csv = _first_named(csv_files, {"labels.csv", "label.csv"})
    if data_csv is None or labels_csv is None:
        return None
    return DatasetFiles(data_csv=data_csv, labels_csv=labels_csv)


def load_expression_data(
    *,
    data_dir: str | Path = "data",
    data_csv: str | Path | None = None,
    labels_csv: str | Path | None = None,
    download: bool = True,
) -> ExpressionDataset:
    """Carga la matriz de expresion y las etiquetas tumorales."""

    if data_csv is not None or labels_csv is not None:
        if data_csv is None or labels_csv is None:
            raise DatasetError("data_csv y labels_csv deben pasarse juntos.")
        files = DatasetFiles(data_csv=Path(data_csv), labels_csv=Path(labels_csv))
    else:
        files = ensure_dataset(data_dir, download=download)

    if not files.data_csv.exists():
        raise DatasetError(f"No existe el archivo de expresion: {files.data_csv}")
    if not files.labels_csv.exists():
        raise DatasetError(f"No existe el archivo de etiquetas: {files.labels_csv}")

    data_frame = pd.read_csv(files.data_csv)
    labels_frame = pd.read_csv(files.labels_csv)
    return parse_expression_frames(data_frame, labels_frame, files)


def parse_expression_frames(
    data_frame: pd.DataFrame,
    labels_frame: pd.DataFrame,
    files: DatasetFiles | None = None,
) -> ExpressionDataset:
    """Convierte archivos estilo UCI en X, y e identificadores de muestra."""

    if data_frame.empty:
        raise DatasetError("El archivo data.csv esta vacio.")
    if labels_frame.empty:
        raise DatasetError("El archivo labels.csv esta vacio.")

    X_raw, data_ids = _drop_identifier_column(data_frame)
    y, label_ids = _extract_labels(labels_frame)

    if len(X_raw) != len(y):
        raise DatasetError(
            "La cantidad de muestras no coincide entre expresion "
            f"({len(X_raw)}) y etiquetas ({len(y)})."
        )

    if data_ids is not None and label_ids is not None:
        if list(data_ids.astype(str)) != list(label_ids.astype(str)):
            raise DatasetError("Los identificadores de muestra no coinciden entre data.csv y labels.csv.")

    sample_ids = data_ids if data_ids is not None else label_ids
    if sample_ids is None:
        sample_ids = pd.Index([f"sample_{index:04d}" for index in range(len(X_raw))], name="sample_id")
    else:
        sample_ids = pd.Index(sample_ids.astype(str), name="sample_id")

    X = X_raw.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        bad_columns = X.columns[X.isna().any()].tolist()[:5]
        raise DatasetError(
            "La matriz de expresion contiene valores faltantes o no numericos. "
            f"Columnas afectadas (primeras): {bad_columns}"
        )

    X.index = sample_ids
    y = pd.Series(y.astype(str).to_numpy(), index=sample_ids, name="tipo_tumor")
    dataset = ExpressionDataset(
        X=X,
        y=y,
        sample_ids=sample_ids,
        data_csv=files.data_csv if files else None,
        labels_csv=files.labels_csv if files else None,
    )
    validate_expression_dataset(dataset)
    return dataset


def validate_expression_dataset(
    dataset: ExpressionDataset,
    *,
    strict_uci: bool = False,
) -> dict[str, object]:
    """Valida estructura minima y devuelve un resumen serializable."""

    X = dataset.X
    y = dataset.y
    if X.empty:
        raise DatasetError("La matriz de expresion no tiene datos.")
    if len(X) != len(y):
        raise DatasetError("X e y deben tener la misma cantidad de muestras.")
    if X.index.has_duplicates:
        raise DatasetError("Los identificadores de muestra estan duplicados.")
    if y.isna().any():
        raise DatasetError("Las etiquetas contienen valores faltantes.")
    if X.isna().any().any():
        raise DatasetError("La matriz de expresion contiene valores faltantes.")

    non_numeric = [column for column in X.columns if not is_numeric_dtype(X[column])]
    if non_numeric:
        raise DatasetError(f"Columnas no numericas en X: {non_numeric[:5]}")

    classes = sorted(y.astype(str).unique().tolist())
    if len(classes) < 2:
        raise DatasetError("Se requieren al menos dos clases tumorales para el analisis.")

    if strict_uci:
        if X.shape != EXPECTED_UCI_SHAPE:
            raise DatasetError(f"Forma inesperada para UCI: {X.shape}; esperado {EXPECTED_UCI_SHAPE}.")
        if set(classes) != EXPECTED_UCI_CLASSES:
            raise DatasetError(f"Clases inesperadas para UCI: {classes}")

    return {
        "n_samples": int(X.shape[0]),
        "n_genes": int(X.shape[1]),
        "classes": classes,
        "class_counts": {str(key): int(value) for key, value in y.value_counts().sort_index().items()},
        "missing_values": 0,
    }


def _drop_identifier_column(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index | None]:
    first_column = frame.columns[0]
    first_name = str(first_column).lower()
    first_values = frame[first_column]
    looks_like_id = first_name.startswith("unnamed") or first_name in {"id", "sample", "sample_id"}
    if looks_like_id or not is_numeric_dtype(first_values):
        return frame.drop(columns=[first_column]), pd.Index(first_values.astype(str), name="sample_id")
    return frame, None


def _extract_labels(labels_frame: pd.DataFrame) -> tuple[pd.Series, pd.Index | None]:
    label_ids: pd.Index | None = None
    frame = labels_frame.copy()
    first_column = frame.columns[0]
    first_name = str(first_column).lower()
    if first_name.startswith("unnamed") or first_name in {"id", "sample", "sample_id"}:
        label_ids = pd.Index(frame[first_column].astype(str), name="sample_id")
        frame = frame.drop(columns=[first_column])

    if frame.empty:
        raise DatasetError("No se encontro una columna de etiquetas en labels.csv.")

    preferred = [column for column in frame.columns if str(column).lower() in {"class", "label", "tipo_tumor"}]
    label_column = preferred[0] if preferred else frame.columns[-1]
    return frame[label_column], label_ids


def _is_supported_archive(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".zip") or name.endswith(".tar.gz") or name.endswith(".tgz") or name.endswith(".tar")


def _extract_one_archive(archive_path: Path, destination: Path) -> list[Path]:
    if zipfile.is_zipfile(archive_path):
        return _extract_zip(archive_path, destination)
    if tarfile.is_tarfile(archive_path):
        return _extract_tar(archive_path, destination)
    return []


def _extract_zip(archive_path: Path, destination: Path) -> list[Path]:
    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = _safe_target(destination, member.filename)
            archive.extract(member, destination)
            extracted.append(target)
    return extracted


def _extract_tar(archive_path: Path, destination: Path) -> list[Path]:
    extracted: list[Path] = []
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            target = _safe_target(destination, member.name)
            archive.extract(member, destination)
            extracted.append(target)
    return extracted


def _safe_target(destination: Path, member_name: str) -> Path:
    target = (destination / member_name).resolve()
    destination_resolved = destination.resolve()
    if destination_resolved != target and destination_resolved not in target.parents:
        raise DatasetError(f"Ruta insegura en archivo comprimido: {member_name}")
    return target


def _first_named(paths: Iterable[Path], names: set[str]) -> Path | None:
    for path in paths:
        if path.name.lower() in names:
            return path
    return None

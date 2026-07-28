"""Datos: construcción de manifiestos, split POR PACIENTE, Dataset y transforms.

Punto clave del proyecto (ver README, sección "trampas"): el split se hace por
paciente, nunca por imagen. Todo el resto del pipeline consume un único
manifiesto CSV con columnas normalizadas, sea cual sea el dataset de origen:

    image_path, patient_id, label, source, view, sex, age, split

`label` es binaria: 0 = NORMAL / sin opacidad, 1 = NEUMONÍA / opacidad pulmonar.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

log = logging.getLogger("cxr.data")

MANIFEST_COLUMNS = ["image_path", "patient_id", "label", "source", "view", "sex", "age"]
LABEL_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}


# --------------------------------------------------------------------------- #
# Constructores de manifiesto por dataset
# --------------------------------------------------------------------------- #
def build_manifest_rsna(root: str | Path, exclude_not_normal: bool = False) -> pd.DataFrame:
    """RSNA Pneumonia Detection Challenge (Kaggle).

    Estructura esperada tras descomprimir:
        root/stage_2_train_images/*.dcm
        root/stage_2_train_labels.csv          (patientId, x, y, width, height, Target)
        root/stage_2_detailed_class_info.csv   (patientId, class)

    `Target == 1` equivale a la clase "Lung Opacity". La clase intermedia
    "No Lung Opacity / Not Normal" es radiológicamente anormal pero sin opacidad;
    por defecto se conserva como negativa (problema más realista y más difícil).
    Con `exclude_not_normal=True` se elimina, dejando Normal vs Lung Opacity.
    """
    root = Path(root)
    labels_csv = root / "stage_2_train_labels.csv"
    detail_csv = root / "stage_2_detailed_class_info.csv"
    images_dir = root / "stage_2_train_images"
    for p in (labels_csv, images_dir):
        if not p.exists():
            raise FileNotFoundError(f"No encuentro {p}. ¿Descargaste y descomprimiste RSNA en {root}?")

    labels = pd.read_csv(labels_csv)[["patientId", "Target"]].drop_duplicates("patientId")
    if detail_csv.exists():
        detail = pd.read_csv(detail_csv).drop_duplicates("patientId")
        labels = labels.merge(detail, on="patientId", how="left")
        if exclude_not_normal:
            before = len(labels)
            labels = labels[labels["class"] != "No Lung Opacity / Not Normal"]
            log.info("Excluida la clase ambigua 'No Lung Opacity / Not Normal': %d -> %d",
                     before, len(labels))

    df = pd.DataFrame(
        {
            "image_path": [str(images_dir / f"{pid}.dcm") for pid in labels["patientId"]],
            # En RSNA el patientId es único por imagen: el agrupamiento es inocuo
            # aquí, pero mantiene el pipeline correcto para datasets con varias
            # radiografías por paciente (NIH, CheXpert...).
            "patient_id": labels["patientId"].astype(str).values,
            "label": labels["Target"].astype(int).values,
            "source": "rsna",
        }
    )
    df = _attach_dicom_metadata(df)
    return _finalize_manifest(df)


def build_manifest_kaggle_pneumonia(root: str | Path) -> pd.DataFrame:
    """"Chest X-Ray Images (Pneumonia)" (paultimothymooney/chest-xray-pneumonia).

    Estructura esperada:
        root/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg

    Ojo: los splits originales están mal repartidos (val = 16 imágenes) y no
    garantizan separación por paciente, así que se ignoran y se rehacen aquí.
    El identificador de paciente se deriva del nombre de fichero:
        person1_bacteria_1.jpeg  -> "person1"
        NORMAL2-IM-1440-0001.jpeg -> "NORMAL2-IM-1440"
    """
    root = Path(root)
    base = root / "chest_xray" if (root / "chest_xray").exists() else root
    files = [p for p in base.rglob("*") if p.suffix.lower() in {".jpeg", ".jpg", ".png"}]
    if not files:
        raise FileNotFoundError(f"No encuentro imágenes bajo {base}.")

    rows = []
    for p in files:
        parent = p.parent.name.upper()
        if parent not in {"NORMAL", "PNEUMONIA"}:
            continue
        rows.append(
            {
                "image_path": str(p),
                "patient_id": _patient_id_from_filename(p.stem),
                "label": 1 if parent == "PNEUMONIA" else 0,
                "source": "kaggle_pneumonia",
                "view": "UNKNOWN",
                "sex": "UNKNOWN",
                "age": np.nan,
            }
        )
    df = pd.DataFrame(rows)
    log.info("Kaggle pneumonia: %d imágenes, %d pacientes", len(df), df["patient_id"].nunique())
    return _finalize_manifest(df)


def build_manifest_folder(root: str | Path) -> pd.DataFrame:
    """Fallback genérico: root/<CLASE>/*.png con CLASE en {NORMAL, PNEUMONIA}."""
    root = Path(root)
    rows = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".dcm"}:
            continue
        parent = p.parent.name.upper()
        if parent not in {"NORMAL", "PNEUMONIA"}:
            continue
        rows.append(
            {
                "image_path": str(p),
                "patient_id": _patient_id_from_filename(p.stem),
                "label": 1 if parent == "PNEUMONIA" else 0,
                "source": "folder",
                "view": "UNKNOWN",
                "sex": "UNKNOWN",
                "age": np.nan,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No encuentro imágenes con estructura <CLASE>/ bajo {root}.")
    return _finalize_manifest(pd.DataFrame(rows))


MANIFEST_BUILDERS: dict[str, Callable[..., pd.DataFrame]] = {
    "rsna": build_manifest_rsna,
    "kaggle_pneumonia": build_manifest_kaggle_pneumonia,
    "folder": build_manifest_folder,
}


def _patient_id_from_filename(stem: str) -> str:
    m = re.match(r"(person\d+)", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # NORMAL2-IM-1440-0001 -> NORMAL2-IM-1440 (quita el índice de imagen final)
    m = re.match(r"(.+)-\d+$", stem)
    if m:
        return m.group(1)
    return stem


def _attach_dicom_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Lee sexo, edad y proyección (AP/PA) de las cabeceras DICOM.

    Son metadatos ya de-identificados en los datasets públicos, y habilitan el
    análisis de subgrupos (fairness) y el control del sesgo AP/PA.
    """
    try:
        import pydicom
    except ImportError:
        log.warning("pydicom no instalado: no se leen metadatos DICOM.")
        df["view"], df["sex"], df["age"] = "UNKNOWN", "UNKNOWN", np.nan
        return df

    from tqdm.auto import tqdm

    views, sexes, ages = [], [], []
    for path in tqdm(df["image_path"], desc="Leyendo cabeceras DICOM", unit="img"):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            views.append(str(getattr(ds, "ViewPosition", "UNKNOWN") or "UNKNOWN"))
            sexes.append(str(getattr(ds, "PatientSex", "UNKNOWN") or "UNKNOWN"))
            raw_age = str(getattr(ds, "PatientAge", "") or "")
            ages.append(float(re.sub(r"\D", "", raw_age)) if re.sub(r"\D", "", raw_age) else np.nan)
        except Exception:  # cabecera corrupta o ausente: no bloquea el pipeline
            views.append("UNKNOWN")
            sexes.append("UNKNOWN")
            ages.append(np.nan)
    df["view"], df["sex"], df["age"] = views, sexes, ages
    return df


def _finalize_manifest(df: pd.DataFrame) -> pd.DataFrame:
    for col in MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = "UNKNOWN" if col in {"view", "sex", "source"} else np.nan
    df = df[MANIFEST_COLUMNS].copy()
    df["label"] = df["label"].astype(int)
    df["patient_id"] = df["patient_id"].astype(str)
    return df.sort_values("image_path").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Split por paciente
# --------------------------------------------------------------------------- #
def make_splits(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Añade la columna `split` (train/val/test) agrupando POR PACIENTE.

    Usa StratifiedGroupKFold para mantener la prevalencia de la clase positiva
    en los tres subconjuntos sin que un mismo paciente cruce la frontera.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    df = df.copy()

    def _holdout(frame: pd.DataFrame, frac: float, rs: int) -> np.ndarray:
        """Devuelve el índice posicional del holdout de tamaño ~frac."""
        n_splits = max(2, int(round(1.0 / frac)))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rs)
        _, hold_idx = next(sgkf.split(frame, frame["label"].values, frame["patient_id"].values))
        return hold_idx

    test_pos = _holdout(df, test_size, seed)
    test_patients = set(df.iloc[test_pos]["patient_id"])

    rest = df[~df["patient_id"].isin(test_patients)].reset_index(drop=True)
    val_frac_of_rest = val_size / max(1e-9, (1.0 - test_size))
    val_pos = _holdout(rest, val_frac_of_rest, seed + 1)
    val_patients = set(rest.iloc[val_pos]["patient_id"])

    df["split"] = "train"
    df.loc[df["patient_id"].isin(val_patients), "split"] = "val"
    df.loc[df["patient_id"].isin(test_patients), "split"] = "test"

    assert_no_patient_leakage(df)
    log.info("Split por paciente:\n%s", split_summary(df).to_string())
    return df


def assert_no_patient_leakage(df: pd.DataFrame) -> None:
    """Falla ruidosamente si un paciente aparece en más de un split."""
    per_patient = df.groupby("patient_id")["split"].nunique()
    offenders = per_patient[per_patient > 1]
    if len(offenders):
        raise AssertionError(
            f"FUGA DE DATOS: {len(offenders)} pacientes aparecen en más de un split "
            f"(p.ej. {list(offenders.index[:5])})."
        )


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("split").agg(
        imagenes=("image_path", "count"),
        pacientes=("patient_id", "nunique"),
        positivos=("label", "sum"),
    )
    g["prevalencia"] = (g["positivos"] / g["imagenes"]).round(4)
    return g.loc[[s for s in ["train", "val", "test"] if s in g.index]]


# --------------------------------------------------------------------------- #
# Lectura de imágenes
# --------------------------------------------------------------------------- #
def load_image(path: str | Path) -> np.ndarray:
    """Devuelve la radiografía como uint8 RGB (H, W, 3).

    Gestiona DICOM (incluida la inversión de gris de MONOCHROME1 y el
    RescaleSlope/Intercept) y formatos estándar PNG/JPEG.
    """
    path = Path(path)
    if path.suffix.lower() == ".dcm":
        import pydicom

        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
        arr = arr * slope + intercept
        # MONOCHROME1: el blanco es el valor mínimo -> hay que invertir
        if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
            arr = arr.max() - arr
        arr = _to_uint8(arr)
    else:
        from PIL import Image

        arr = np.array(Image.open(path).convert("L"))

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return np.ascontiguousarray(arr.astype(np.uint8))


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    return (((arr - lo) / (hi - lo)) * 255.0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int = 224, train: bool = False, use_clahe: bool = False):
    """Augmentation apropiada para tórax.

    NUNCA se usa flip horizontal: invertiría la lateralidad anatómica
    (corazón a la izquierda del paciente) y enseñaría al modelo una anatomía
    de situs inversus que no existe en la práctica. Tampoco se usan
    deformaciones agresivas que puedan borrar hallazgos sutiles.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    tail = [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]

    if not train:
        return A.Compose([A.Resize(img_size, img_size), *tail])

    aug = [A.Resize(img_size, img_size)]
    if use_clahe:
        aug.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3))
    aug += [
        A.Affine(scale=(0.92, 1.08), translate_percent=(-0.05, 0.05), rotate=(-10, 10), p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    ]
    return A.Compose([*aug, *tail])


# --------------------------------------------------------------------------- #
# Dataset y DataLoaders
# --------------------------------------------------------------------------- #
try:  # permite construir manifiestos en un entorno sin torch
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # pragma: no cover
    _TorchDataset = object  # type: ignore[assignment,misc]


class ChestXrayDataset(_TorchDataset):  # type: ignore[misc]
    """Dataset a partir del manifiesto normalizado."""

    def __init__(self, df: pd.DataFrame, transform=None, return_meta: bool = False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        import torch

        row = self.df.iloc[idx]
        image = load_image(row["image_path"])
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        if self.return_meta:
            return image, label, {"image_path": row["image_path"], "patient_id": row["patient_id"]}
        return image, label


def make_loaders(
    manifest: pd.DataFrame,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    use_clahe: bool = False,
    balance_train: bool = False,
):
    """Crea los DataLoaders de train/val/test desde el manifiesto ya splitteado."""
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    loaders = {}
    for split in ["train", "val", "test"]:
        part = manifest[manifest["split"] == split]
        if part.empty:
            continue
        is_train = split == "train"
        ds = ChestXrayDataset(part, build_transforms(img_size, is_train, use_clahe))

        sampler = None
        if is_train and balance_train:
            counts = part["label"].value_counts().to_dict()
            weights = part["label"].map(lambda y: 1.0 / counts[y]).values
            sampler = WeightedRandomSampler(
                torch.as_tensor(weights, dtype=torch.double), num_samples=len(part), replacement=True
            )

        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train and sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=num_workers > 0,
        )
    return loaders


def pos_weight_from(manifest: pd.DataFrame) -> float:
    """pos_weight para BCEWithLogitsLoss = n_negativos / n_positivos en train."""
    train = manifest[manifest["split"] == "train"]
    n_pos = int(train["label"].sum())
    n_neg = int(len(train) - n_pos)
    return float(n_neg / max(1, n_pos))

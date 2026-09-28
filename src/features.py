# src/features.py
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# --- Constantes compartidas ---
COLUMNAS_SENSOR = ["temp_c", "power_w", "util_pct", "clock_mhz"]
VENTANA_SEGUNDOS = 30          # tamaño de ventana en entrenamiento
MIN_LECTURAS_API = 10          # mínimo aceptable en la API

# Orden canónico de features. train y api DEBEN producir este orden.
FEATURE_NAMES = [
    # temp_c
    "temp_c_mean", "temp_c_std", "temp_c_min", "temp_c_max",
    # power_w
    "power_w_mean", "power_w_std", "power_w_min", "power_w_max", "power_w_range",
    # util_pct
    "util_pct_mean", "util_pct_std",
    # clock_mhz
    "clock_mhz_mean", "clock_mhz_std", "clock_mhz_min", "clock_mhz_max",
    # ecc_errors
    "ecc_total", "ecc_max",
]


def _features_de_ventana(ventana: pd.DataFrame) -> dict:
    """Calcula features de UNA ventana (ya sin episodio_id/estado)."""
    f: dict[str, float] = {}

    for col in COLUMNAS_SENSOR:
        f[f"{col}_mean"] = float(ventana[col].mean())
        f[f"{col}_std"] = float(ventana[col].std(ddof=0))
        f[f"{col}_min"] = float(ventana[col].min())
        f[f"{col}_max"] = float(ventana[col].max())

    # rango de potencia: delata falla_alimentacion (bajones/picos)
    f["power_w_range"] = f["power_w_max"] - f["power_w_min"]

    # ECC: total y máximo delatan degradacion_memoria
    f["ecc_total"] = float(ventana["ecc_errors"].sum())
    f["ecc_max"] = float(ventana["ecc_errors"].max())

    return f


def features_desde_ventana(lecturas: Iterable[dict]) -> pd.DataFrame:
    """
    API: recibe lista de dicts (una lectura/segundo) y devuelve
    un DataFrame de UNA fila con las features en el orden canónico.
    """
    df = pd.DataFrame(list(lecturas))
    # Validación mínima de columnas
    faltantes = [c for c in COLUMNAS_SENSOR + ["ecc_errors"] if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en la ventana: {faltantes}")

    feats = _features_de_ventana(df)
    # Reordena exactamente como en entrenamiento
    return pd.DataFrame([feats])[FEATURE_NAMES]


def features_desde_dataframe(df: pd.DataFrame,
                             ventana: int = VENTANA_SEGUNDOS) -> pd.DataFrame:
    """
    TRAIN: parte el dataset por episodio_id en ventanas no solapadas.
    Cada ventana hereda el estado de su episodio.
    Devuelve X (features) + y (estado).
    """
    registros = []
    for ep_id, grupo in df.groupby("episodio_id", sort=True):
        grupo = grupo.sort_values("segundo").reset_index(drop=True)
        n = len(grupo)
        for inicio in range(0, n - ventana + 1, ventana):
            trozo = grupo.iloc[inicio:inicio + ventana]
            feats = _features_de_ventana(trozo)
            feats["estado"] = trozo["estado"].iloc[0]     # etiqueta heredada
            feats["episodio_id"] = ep_id                  # solo para GroupKFold
            registros.append(feats)

    out = pd.DataFrame(registros)
    return out[FEATURE_NAMES + ["estado", "episodio_id"]]
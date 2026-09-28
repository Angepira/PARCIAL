# src/train.py
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Permite `python -m src.train` desde la raíz
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.features import FEATURE_NAMES, features_desde_dataframe
from src.schema import validar_telemetria

RAIZ = Path(__file__).resolve().parent.parent
CSV = RAIZ / "data" / "telemetria_publica.csv"
MODELO = RAIZ / "models" / "modelo.joblib"


def _verificar_episodios_puros(df: pd.DataFrame) -> None:
    """Asegura que cada episodio_id tiene un único estado (contrato del parcial)."""
    conteo = df.groupby("episodio_id")["estado"].nunique()
    sucios = conteo[conteo > 1]
    if not sucios.empty:
        raise ValueError(
            f"Episodios con más de un estado (violan el contrato): {sucios.to_dict()}"
        )


def main() -> None:
    print("Cargando datos...")
    df = pd.read_csv(CSV)
    print(f"Filas crudas: {len(df)}")

    print("Validando con pandera (descarta filas inválidas)...")
    df_limpio = validar_telemetria(df)
    print(f"Filas válidas tras contrato: {len(df_limpio)} (descartadas: {len(df) - len(df_limpio)})")

    _verificar_episodios_puros(df_limpio)
    print("Episodios puros: OK")

    print("Ventaneando y calculando features...")
    dataset = features_desde_dataframe(df_limpio, ventana=30)
    print(f"Ventanas: {len(dataset)}")

    X = dataset[FEATURE_NAMES]
    y = dataset["estado"]
    grupos = dataset["episodio_id"]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    cv = GroupKFold(n_splits=4)
    scores = cross_val_score(pipe, X, y, groups=grupos, cv=cv, scoring="accuracy")
    print(f"Accuracy CV (GroupKFold por episodio): {scores.mean():.3f} ± {scores.std():.3f}")

    pipe.fit(X, y)
    MODELO.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "feature_names": FEATURE_NAMES}, MODELO)
    print(f"Modelo guardado en {MODELO}")


if __name__ == "__main__":
    main()
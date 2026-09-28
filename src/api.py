# src/api.py
from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.features import (
    FEATURE_NAMES,
    MIN_LECTURAS_API,
    features_desde_ventana,
)

RAIZ = Path(__file__).resolve().parent.parent
MODELO_PATH = RAIZ / "models" / "modelo.joblib"

app = FastAPI(title="Detector de fallas GPU", version="1.0.0")

# Carga del modelo al arrancar (NO se entrena aquí)
_artefacto = joblib.load(MODELO_PATH)
_pipeline = _artefacto["pipeline"]
_feature_names = _artefacto["feature_names"]
assert _feature_names == FEATURE_NAMES, "El modelo fue entrenado con otras features"


# ---------- Esquemas de entrada/salida (validación en la puerta) ----------
class Lectura(BaseModel):
    temp_c: float = Field(..., ge=-20, le=120, description="Temperatura en °C")
    power_w: float = Field(..., ge=0, le=1000, description="Potencia en W")
    util_pct: float = Field(..., ge=0, le=100, description="Utilización %")
    clock_mhz: float = Field(..., ge=0, le=4000, description="Reloj MHz")
    ecc_errors: int = Field(..., ge=0, description="Errores ECC del segundo")


class PeticionVentana(BaseModel):
    lecturas: List[Lectura] = Field(..., min_length=MIN_LECTURAS_API)

    @field_validator("lecturas")
    @classmethod
    def tamano_razonable(cls, v: List[Lectura]) -> List[Lectura]:
        if len(v) < MIN_LECTURAS_API:
            raise ValueError(
                f"Se requieren al menos {MIN_LECTURAS_API} lecturas "
                f"para calcular features estables (recibidas: {len(v)})"
            )
        return v


class RespuestaPrediccion(BaseModel):
    estado_predicho: str
    confianza: float


# ---------- Endpoints ----------
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "features": _feature_names}


@app.post("/predecir", response_model=RespuestaPrediccion)
def predecir(peticion: PeticionVentana) -> RespuestaPrediccion:
    try:
        lecturas = [lec.model_dump() for lec in peticion.lecturas]
        X = features_desde_ventana(lecturas)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Ventana inválida: {e}")

    pred = _pipeline.predict(X)[0]
    proba = _pipeline.predict_proba(X)[0]
    confianza = float(max(proba))
    return RespuestaPrediccion(estado_predicho=str(pred), confianza=confianza)
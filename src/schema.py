# src/schema.py
import pandera as pa
import pandas as pd
from pandera.typing import Series

ESTADOS_VALIDOS = ["normal", "sobrecalentamiento", "degradacion_memoria", "falla_alimentacion"]


class TelemetriaSchema(pa.DataFrameModel):
    """Contrato de la telemetría cruda (una fila = un segundo)."""

    episodio_id: Series[int] = pa.Field(ge=0)
    segundo: Series[int] = pa.Field(ge=0)
    temp_c: Series[float] = pa.Field(ge=-20, le=120)
    power_w: Series[float] = pa.Field(ge=0, le=1000)
    util_pct: Series[float] = pa.Field(ge=0, le=100)
    clock_mhz: Series[float] = pa.Field(ge=0, le=4000)
    ecc_errors: Series[int] = pa.Field(ge=0)
    estado: Series[str] = pa.Field(isin=ESTADOS_VALIDOS)

    class Config:
        strict = True
        coerce = True
        drop_invalid_rows = True   # ← clave: en vez de lanzar, descarta esas filas


def validar_telemetria(df: pd.DataFrame) -> pd.DataFrame:
    """Valida y devuelve solo las filas que cumplen el contrato."""
    return TelemetriaSchema.validate(df, lazy=True)
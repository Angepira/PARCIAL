# Detector de fallas GPU — NVIDIA L40

Servicio de clasificación de telemetría de GPU en 4 estados:
`normal`, `sobrecalentamiento`, `degradacion_memoria`, `falla_alimentacion`.

## Estructura

```
detector-fallas-gpu/
├── src/
│   ├── features.py   # ventaneo + features (usado por train y api)
│   ├── schema.py     # contrato pandera
│   ├── train.py      # entrena y serializa
│   └── api.py        # FastAPI
├── models/modelo.joblib
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

## Entrenar (una sola vez, fuera del contenedor)

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train
```
Esto produce `models/modelo.joblib`. **No se entrena al arrancar el contenedor.**

## Construir y correr

```bash
docker build -t detector-gpu .
docker run -p 8000:8000 detector-gpu
```

Docs interactivas: http://localhost:8000/docs

## Ejemplo de petición

```bash
curl -X POST http://localhost:8000/predecir \
  -H "Content-Type: application/json" \
  -d '{"lecturas": [
    {"temp_c":88.3,"power_w":296.5,"util_pct":95.8,"clock_mhz":2038,"ecc_errors":0},
    {"temp_c":89.8,"power_w":298.9,"util_pct":95.4,"clock_mhz":2091,"ecc_errors":1},
    {"temp_c":84.6,"power_w":284.7,"util_pct":91.8,"clock_mhz":2043,"ecc_errors":0},
    {"temp_c":89.2,"power_w":297.3,"util_pct":92.3,"clock_mhz":2053,"ecc_errors":0},
    {"temp_c":85.8,"power_w":288.9,"util_pct":92.2,"clock_mhz":2305,"ecc_errors":0},
    {"temp_c":87.6,"power_w":301.5,"util_pct":88.6,"clock_mhz":2080,"ecc_errors":0},
    {"temp_c":93.1,"power_w":310.3,"util_pct":95.6,"clock_mhz":2122,"ecc_errors":0},
    {"temp_c":83.8,"power_w":279.1,"util_pct":94.8,"clock_mhz":2119,"ecc_errors":0},
    {"temp_c":87.2,"power_w":297.7,"util_pct":98.0,"clock_mhz":2147,"ecc_errors":0},
    {"temp_c":84.2,"power_w":303.0,"util_pct":100.0,"clock_mhz":2062,"ecc_errors":0}
  ]}'
```

Respuesta:
```json
{"estado_predicho":"sobrecalentamiento","confianza":0.98}
```

## Decisiones de diseño

- **Features compartidas** entre `train` y `api` vía `src/features.py`.
- **Ventanas de 30 s por episodio**, sin solapamiento ni cruce de episodios.
- **Validación doble**: pandera al entrenar, Pydantic en la API.
- **Mínimo 10 lecturas** por ventana (si no, 422).
- **Modelo serializado dentro de la imagen**, no se entrena al arrancar.
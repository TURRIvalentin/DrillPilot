# M7: Backend FastAPI

## Decisiones previas (prerequisito de esta milestone)

Antes de escribir los endpoints se resolvieron dos decisiones de contrato:

1. **Arquitectura de latencia** — [docs/adr/005-shap-endpoint-design.md](adr/005-shap-endpoint-design.md):
   `/predict` nunca calcula SHAP (p50=4.53ms/p95=5.80ms medido en M5), `/explain` es
   un endpoint separado que sí lo hace (p50=14.20ms/p95=21.02ms, ~3.1x el costo base) y
   solo explica la última lectura de la ventana recibida.
2. **Confidence flag** — `known_limitation_zone: bool` en la respuesta de ambos
   endpoints, `True` cuando `MD` cae en `ml.evaluation.metrics.KNOWN_LIMITATION_MD_RANGE_M`
   (634.0–988.0 m, límites inclusivos), la constante ya centralizada como fuente única
   (antes duplicada como `REGIME_GAP_MD_RANGE_M` en `ml/explainability/shap_explain.py`).
   Documentado y confirmado tres veces de forma independiente (router de M4, SHAP de
   M5, inferencia en vivo de M6) — ver [docs/m6_results.md](m6_results.md).

## Qué se construyó

```
backend/app/
  core/
    config.py       Settings minimo (model_uri, log_level via env vars)
    logging.py       configure_logging(), llamado desde el lifespan de main.py
    exceptions.py    ValueError->422, ModelNotLoadedError->503, Exception->500
  schemas/
    reading.py        Reading: 1 fila de la ventana, campos 1:1 con
                       ml.features.pipeline.DIRECT_FEATURE_COLUMNS + well_id
    predict.py         PredictionRequest/PredictionItem/PredictionResponse
    explain.py          ExplainRequest (= PredictionRequest), FeatureContribution,
                        ExplainResponse
    health.py           HealthResponse
  services/
    inference_service.py  InferenceService: unica clase que envuelve
                           ml.inference.predict.predict_rop (para /predict) y
                           shap.TreeExplainer sobre pipeline.named_steps["model"]
                           (para /explain, construido lazy en el primer llamado)
  api/
    deps.py    get_inference_service() -- lee la instancia unica desde app.state
    predict.py  POST /predict
    explain.py   POST /explain
    health.py     GET /health
  main.py   FastAPI app; el lifespan carga el pipeline de produccion UNA vez
            (mlflow.sklearn.load_model, alias "production") y lo guarda en
            app.state.inference_service; si la carga falla el proceso no crashea --
            /health reporta model_loaded=false y /predict, /explain devuelven 503
            hasta que el registry este disponible.
```

Ningún módulo de `backend/` reimplementa lógica de features o de modelo:
`InferenceService.predict()` llama a `ml.inference.predict.predict_rop` sin
modificarla, y `InferenceService.explain()` reusa
`pipeline.named_steps["features"]`/`["model"]` (el mismo artefacto combinado
construido en M6 por `ml/training/promote_model.py`), en vez de reconstruir el
pipeline o el modelo por su cuenta.

## Contrato de entrada/salida

- **Request** (`/predict` y `/explain`, mismo shape): `{"readings": [Reading, ...]}`,
  mínimo 1 fila, ordenadas por `MD` ascendente dentro de un mismo `well_id` (ADR-004).
  Una fila con `MD` no creciente dentro de su `well_id` es rechazada con 422 (el
  `ValueError` que ya lanza `USROPFeatureTransformer._validate_input`, mapeado por
  `backend/app/core/exceptions.py`).
- **`/predict`** devuelve una `PredictionItem` por fila recibida:
  `{md, predicted_rop, known_limitation_zone}`.
- **`/explain`** devuelve una sola explicación, de la última fila de la ventana:
  `{md, predicted_rop, known_limitation_zone, base_value, contributions[]}`, donde
  `contributions` trae `{feature, value, shap_value}` para las 14 features del
  pipeline. La propiedad de aditividad de SHAP
  (`sum(shap_value) + base_value == predicted_rop`) se verifica explícitamente en
  `tests/backend/test_explain.py::test_explain_satisfies_shap_additivity`.
- **`/health`** devuelve `{status, model_loaded, model_uri}` -- `status="degraded"`
  si el modelo no pudo cargarse al arrancar.

## Pruebas manuales contra el modelo real

Con `uvicorn backend.app.main:app` levantado, el `/health` inicial confirmó carga
real desde el registry:

```json
{"status":"ok","model_loaded":true,"model_uri":"models:/drillpilot-rop@production"}
```

`/predict` con 15 lecturas reales del pozo 3 (MD 1306.5–1309.3 m, fuera de la banda
de limitación conocida) devolvió 15 predicciones con `known_limitation_zone=false`
en todas, y `/explain` sobre la misma ventana devolvió la explicación de la última
fila (MD=1309.268) con 14 contribuciones SHAP.

Para confirmar el flag en datos reales (no solo sintéticos), se pidieron 5 lecturas
del pozo 0 con `MD` entre 634.0 y 988.0 (el extremo inferior exacto de la banda,
634.011–634.222 m): las 5 respuestas trajeron `known_limitation_zone=true`. No hay
en `docs/m6_results.md` un MAE medido exactamente en este sub-tramo (la tabla ahí
reporta por tramos de filas, no por el rango MD 634-988m puntual); el MAE~31.02
documentado para el pozo 0 corresponde al tramo final MD≥1,114m, que está pasando
la banda, no dentro de ella -- se cita como evidencia de que el error se dispara
más allá de este punto, no como el número exacto en 634m.

## Tests

`tests/backend/` (19 tests, todos pasando):

- `test_health.py` (2): modelo cargado -> `status=ok`; no cargado -> `status=degraded`.
- `test_predict_endpoint.py` (6): una predicción por fila, ventana de 1 fila
  aceptada, lista vacía rechazada (422), columna faltante rechazada (422), MD no
  ascendente rechazado (422), modelo no cargado -> 503.
- `test_explain.py` (5): una contribución por feature (14), aditividad SHAP exacta
  (`abs=1e-6`), explica solo la última fila de la ventana, lista vacía rechazada,
  modelo no cargado -> 503.
- `test_known_limitation_zone.py` (6): el test dedicado que pide el enunciado de
  M7 -- confirma que el flag se activa dentro de la banda (`/predict` y
  `/explain`), no se activa fuera, los dos límites (634.0 y 988.0) son inclusive, y
  el flag se apaga justo pasado el límite superior (988.1).

Fixtures (`tests/backend/conftest.py`): sin acceso a MLflow ni red -- cada test
inyecta un pipeline pequeño y ya ajustado vía `app.dependency_overrides`, mismo
patrón que `tests/ml/test_predict.py`. `/predict` y `/health` usan
`GlobalMeanBaseline` (rápido); `/explain` necesita un modelo de árbol real
(`shap.TreeExplainer` no soporta `GlobalMeanBaseline`), así que usa un LightGBM
diminuto (`n_estimators=5`) ajustado sobre 20 filas sintéticas.

Suite completa del proyecto: **80/80 tests pasando** (61 en `tests/ml/` + 19 en
`tests/backend/`). 9/9 hooks de pre-commit pasando (`trailing-whitespace`,
`end-of-file-fixer`, `check-yaml`, `check-toml`, `check-added-large-files`,
`check-merge-conflict`, `ruff`, `ruff-format`, `mypy`).

## Decisiones menores durante la implementación

- **`fastapi`/`uvicorn[standard]`/`pydantic>=2`** instalados en `.venv` desde el
  grupo `api` ya declarado en `pyproject.toml` desde M0 (nunca instalado hasta
  ahora). Se agregó `httpx` al grupo `dev` -- es una dependencia solo de testing
  (`fastapi.testclient.TestClient` la requiere), no del runtime de la API.
- **Sin `pydantic-settings`**: `backend/app/core/config.py` es una clase mínima que
  lee `os.environ` directamente. El único valor configurable hoy es `model_uri`;
  agregar una dependencia nueva para una sola variable de entorno sería
  sobre-ingeniería para lo que M7 necesita.
- **`ruff`'s B008 (`Depends` en default de argumento)**: falso positivo conocido
  para el patrón idiomático de FastAPI. Se resolvió agregando
  `fastapi.Depends` a `tool.ruff.lint.flake8-bugbear.extend-immutable-calls` en
  `pyproject.toml` -- la solución que la propia documentación de ruff recomienda
  para proyectos FastAPI, en vez de reescribir el patrón estándar del framework.
- **Colisión de nombre de módulo de test**: `tests/backend/test_predict.py`
  colisionaba con `tests/ml/test_predict.py` bajo el modo de importación
  "rootless" de pytest (no hay `__init__.py` en `tests/`, consistente con el resto
  del proyecto). Renombrado a `tests/backend/test_predict_endpoint.py`.
- **`HTTP_422_UNPROCESSABLE_ENTITY`**: deprecado en la versión de Starlette
  instalada a favor de `HTTP_422_UNPROCESSABLE_CONTENT` (mismo código 422);
  actualizado en `backend/app/core/exceptions.py` para no dejar un warning de
  deprecación nuevo en la suite.

## Fuera de alcance (explícitamente, por instrucción del usuario)

Streamlit y Docker quedan para M8+; no se tocó nada de eso en esta milestone.

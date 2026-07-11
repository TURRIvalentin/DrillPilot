# M9-M10: Dockerización + CI/CD

## ADR-006: empaquetado del modelo y target de despliegue

[docs/adr/006-model-packaging-deploy.md](adr/006-model-packaging-deploy.md) --
resumen de las 4 decisiones pedidas:

1. El artefacto se empaqueta dentro de la imagen en build time (`COPY` plano),
   nunca se carga desde MLflow en runtime.
2. Consecuencia explícita: sin hot-swap -- reentrenar implica reconstruir y
   redesplegar.
3. Target: AWS ECS Fargate, no Lambda (tamaño de imagen + cold start con
   lightgbm/shap/mlflow).
4. Referencia explícita: `ml/inference/export_model.py::PINNED_PRODUCTION_RUN_ID`
   (un `run_id`, no el alias `"production"` resuelto en build time).

Un hallazgo no buscado durante esta decisión, documentado en el ADR: el tracking
store local de este proyecto vive en una carpeta *hermana* con nombre
percent-encoded literal (`AI%20Engineer%20Portfolio` en vez de
`AI Engineer Portfolio`, por cómo MLflow arma la URI sqlite en Windows con
espacios en la ruta) -- confirmado con `mlflow.tracking.MlflowClient`,
inspección directa del `SqlAlchemyStore` interno, y verificación con `sqlite3`
puro. No rompe nada en desarrollo local (la resolución es consistente y
reproducible en múltiples corridas), pero es la razón concreta, no solo
preferencia de diseño, por la que el artefacto se exporta a un directorio plano
en vez de depender del tracking store dentro del build de Docker.

También se verificó directamente que copiar tal cual el directorio de artefacto
interno de MLflow (`mlruns/1/models/m-<id>/artifacts/`) falla al cargar con
`mlflow.sklearn.load_model()` sobre una ruta llana sin tracking store, en
Windows:

```
mlflow.exceptions.MlflowException: No such artifact: 'model_artifact_test'
```

mientras que el patrón `mlflow.sklearn.save_model(model, out_dir)` seguido de
`mlflow.sklearn.load_model(out_dir)` sí funciona de punta a punta, sin ningún
tracking store, con las mismas predicciones exactas ya verificadas en M7/M8
(`[10.06956566 10.06956566 9.92444742]` para el pozo 3). Esta es la técnica que
usa `ml/inference/export_model.py`.

## Fix real encontrado en el camino: `InferenceService.load()` ignoraba `settings.model_uri`

Antes de que el empaquetado en Docker pudiera funcionar, hizo falta corregir un
bug real en `backend/app/services/inference_service.py`: `load()` llamaba a
`load_production_model()` sin argumentos, ignorando por completo
`settings.model_uri` (la variable de entorno `DRILLPILOT_MODEL_URI`, ya
declarada desde M7 pero nunca conectada a nada). Un contenedor que seteara
`DRILLPILOT_MODEL_URI=/app/docker/model_artifact` hubiera seguido intentando
resolver el alias `"production"` contra un tracking store inexistente en
runtime. Corregido en `ml/inference/predict.py::load_production_model()`
(ahora acepta `model_uri` como parámetro) y en `InferenceService.load()` (ahora
lo pasa). Verificado end-to-end antes de tocar Docker:

```python
DRILLPILOT_MODEL_URI=docker/model_artifact -> settings.model_uri == 'docker/model_artifact'
-> load_production_model(settings.model_uri) -> Pipeline cargado correctamente
```

## M9: Dockerización

```
Dockerfile              backend, multi-stage (builder con build-essential
                         descartado; runtime solo con libgomp1 + el venv ya
                         armado). Copia docker/model_artifact/ y fija
                         DRILLPILOT_MODEL_URI=/app/docker/model_artifact.
frontend/Dockerfile      frontend, imagen liviana aparte -- nunca copia el
                         artefacto del modelo (habla HTTP con el backend, no
                         carga el modelo, ver ADR-006).
docker-compose.yml       backend + frontend, healthcheck real del backend como
                         condición de arranque del frontend, ./data montado
                         read-only en el frontend para que "cargar ejemplo
                         real" funcione en local sin hornear el dataset
                         licenciado (CC BY-NC-SA) dentro de la imagen.
.dockerignore            excluye .venv/, mlruns/, data/, docs/, tests/, etc.
                         del contexto de build.
```

### `docker/model_artifact/` -- el artefacto commiteado

Exportado una vez con `python -m ml.inference.export_model`
(`run_id=a302711c1cd34d4da4c86235387dc5f8`, el mismo candidato ya en producción
desde M6). **1.9 MB**, 5 archivos (`MLmodel`, `model.pkl`, `conda.yaml`,
`python_env.yaml`, `requirements.txt`). Se commitea al repositorio -- ver
ADR-006, alternativa (b), para por qué (evita necesitar un artifact store/IaC
antes de que el deploy real a AWS lo justifique).

### Healthcheck real

`HEALTHCHECK` del backend no solo pega al puerto -- corre
`urllib.request.urlopen('http://localhost:8000/health')` y falla si
`model_loaded` no es `true`:

```dockerfile
CMD python -c "import json,sys,urllib.request; \
  r=json.load(urllib.request.urlopen('http://localhost:8000/health', timeout=3)); \
  sys.exit(0 if r.get('model_loaded') else 1)"
```

Un contenedor donde el proceso uvicorn está vivo pero el modelo no cargó (ver
`backend/app/main.py`'s lifespan, que no crashea el proceso en ese caso) se
reporta como `unhealthy`, no `healthy`.

### Prueba real: build + run + requests reales, ambos flags

```
docker compose build   -> drillpilot-backend (1.32 GB), drillpilot-frontend (1.62 GB)
docker compose up -d   -> backend: Healthy, frontend: Healthy
```

(Puertos host remapeados a 8010/8511 en `docker-compose.yml` -- 8000/8501 ya
estaban ocupados en esta máquina por otros proyectos Docker corriendo
[`drillq-backend`, `timonel_rag_container`], sin relación con DrillPilot; no se
tocó ninguno de esos contenedores.)

`GET /health` contra el contenedor:

```json
{"status":"ok","model_loaded":true,"model_uri":"/app/docker/model_artifact"}
```

`model_uri` confirma que está sirviendo desde el artefacto local horneado, no
desde el registry. `POST /predict` y `POST /explain` contra el contenedor, con
las mismas 15 lecturas reales del pozo 3 usadas en M7/M8, devolvieron **las
mismas predicciones exactas** (`10.06956565920712`, `9.924447418660733`, ...) --
el modelo empaquetado es bit-a-bit equivalente al servido vía registry.
`POST /predict` con 5 lecturas reales del pozo 0 en la banda 634-988m
(`MD=634.011`) devolvió `known_limitation_zone=true` en las 5 y, al ser <10
filas, `insufficient_history=true` -- ambos flags nuevos de M7 funcionan
igual dentro del contenedor que corriendo con uvicorn directo.

También se corrió el script real de Streamlit (`streamlit.testing.v1.AppTest`)
**dentro del contenedor frontend** (`docker compose exec frontend python -c
...`), hablando con el backend por la red interna de Docker
(`http://backend:8000`, no localhost) -- carga la ventana de ejemplo real (vía
el volumen `./data` montado), predice, explica, y muestra ambos warnings
combinados cuando corresponde (pozo 0, ventana de 3 filas en la banda
634-988m):

```
- Profundidad dentro de la zona de alto error conocida (634-988 m MD) -- ...
- Ventana con menos de 10 lecturas -- ...
```

Idéntico al comportamiento verificado en M8 sin Docker.

### Limitación conocida: tamaño de imagen del frontend

`drillpilot-frontend` (1.62 GB) es más pesada que `drillpilot-backend` (1.32
GB) pese a ser un cliente HTTP delgado -- instala el grupo `ml` completo
(lightgbm, shap, mlflow, optuna) solo para tener `pandas`/`scikit-learn`
disponibles (`ml.features.pipeline` los necesita), porque `pyproject.toml` no
separa un grupo liviano `ml-core` de las dependencias de entrenamiento. No se
resolvió acá -- reestructurar los grupos de dependencias no fue pedido para
este milestone y el frontend igual corre y pasa todas las pruebas; queda
anotado como optimización futura.

## M10: CI/CD

`.github/workflows/ci.yml` -- dos jobs:

1. **test**: `astral-sh/setup-uv` (con cache habilitado, keyed por
   `pyproject.toml`), instala `.[ml,api,frontend,dev]`, corre `ruff check`,
   `ruff format --check`, `mypy backend ml frontend`, `pytest -q` (suite
   completa: `ml/` + `backend/` + `frontend/`).
2. **docker-build** (depende de que `test` pase): buildea las dos imágenes
   (`docker/build-push-action`, `push: false`) con cache de capas de GitHub
   Actions (`type=gha`). No pushea a ningún registry -- fuera de scope de este
   milestone, según lo pedido.

**Verificación honesta -- qué se probó y qué no:** este repositorio todavía no
tiene un remoto de GitHub configurado (`git remote -v` vacío), así que el
workflow **no corrió en GitHub Actions real** -- no hay forma de mostrar un
badge verde genuino todavía. Lo que sí se verificó, exactamente como el
workflow lo ejecutaría:

- Sintaxis del YAML válida (parseada con PyYAML).
- Cada comando del job `test` corrido localmente en secuencia idéntica
  (`ruff check .`, `ruff format --check .`, `mypy backend ml frontend`,
  `pytest -q`) -- **107/107 tests pasando**, mypy y ruff limpios.
- El job `docker-build`: ambas imágenes ya se buildearon y corrieron de verdad
  (sección M9 arriba) con los mismos Dockerfiles que el workflow referencia.

El badge en el README apunta a `<owner>/<repo>` como placeholder -- hay que
reemplazarlo una vez que el repositorio se suba a GitHub (y entonces sí correrá
el workflow real).

### Hallazgo durante M10: el hook de mypy de pre-commit necesitaba un pin de numpy

Al agregar `requests`/`types-requests` (para que mypy resolviera `api_client.py`
del frontend) y `streamlit` (para `st.cache_data`) a los
`additional_dependencies` del hook de mypy, la instalación aislada del hook
resolvió una versión de numpy más nueva (**2.5.1**) que la que efectivamente usa
el proyecto (**2.4.6**, ver `.venv`) -- streamlit trae numpy como dependencia
transitiva sin que el hook la pinee. La diferencia de stubs entre esas dos
versiones de numpy generó **13 errores nuevos y espurios** en `ml/` (`cv.py`,
`predict.py`, `metrics.py`, `shap_explain.py`, `diagnose_cv_gap.py`) que **no
reproducen** contra el entorno real del proyecto
(`mypy backend ml frontend` en `.venv` sigue limpio). Se corrigió pineando
`numpy==2.4.6` explícitamente en `additional_dependencies` -- mismo principio
que la corrección de M7 (dar al hook las dependencias reales del proyecto en
vez de dejarlo resolver algo distinto), extendido acá a pinnear la versión
exacta cuando el paquete transitivo puede resolver a algo distinto de lo que el
proyecto realmente fija.

De paso, se encontró y corrigió un error de tipo genuino y preexistente (no
relacionado con el mismatch de numpy) en `ml/models/byoung_reduced.py:165`
(`np.exp(log_rop)` devolvía `Any` en vez de `ndarray`) -- mismo patrón que ya
se usa en `ml/inference/predict.py::predict_rop`, resuelto envolviendo con
`np.asarray(...)`.

## Tests

- `tests/ml/test_export_model.py` (2 tests nuevos): el round-trip
  `save_portable_artifact` -> `mlflow.sklearn.load_model()` desde una ruta
  llana produce las mismas predicciones que el pipeline original, y
  sobrescribe correctamente un directorio de export previo.
  `load_pinned_run_model`/`export_model_artifact` (la parte que sí toca el
  tracking store local) no tienen test automatizado -- consistente con cómo
  `tests/ml/test_shap_explain.py` tampoco corre contra el registry real, ver
  el docstring del archivo.
- Nada nuevo en `tests/backend/` -- el fix de `InferenceService.load()` no
  cambia ningún comportamiento observable en los tests existentes (siguen
  inyectando el modelo directamente vía `InferenceService(model=...)`,
  bypaseando `load()`).

Suite completa del proyecto: **107/107 tests pasando** (63 en `tests/ml/`
[61 + 2 nuevos] + 24 en `tests/backend/` + 20 en `tests/frontend/`). 9/9 hooks
de pre-commit pasando, verificado con `.mypy_cache/` limpio.

## Fuera de alcance (explícitamente, por instrucción del usuario)

Deploy real a AWS ECS Fargate, Terraform/IaC, credenciales, redes, y push de
las imágenes a un registry -- quedan para un milestone posterior, ahora que
ADR-006 ya fijó el target y la estrategia de empaquetado.

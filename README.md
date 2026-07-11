# DrillPilot

<!-- TODO: reemplazar <owner>/<repo> una vez que el repositorio tenga un remoto
     de GitHub -- ver docs/m9_m10_results.md, sección CI, para el motivo. -->
[![CI](https://github.com/<owner>/<repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<owner>/<repo>/actions/workflows/ci.yml)

Plataforma open source de Machine Learning para predecir el Rate of
Penetration (ROP) durante la perforación de pozos de petróleo, construida
con arquitectura de producción (no un notebook) y foco en explicabilidad
(SHAP) sobre cada predicción.

**Sobre el input real del modelo:** no predice a partir de una lectura
instantánea aislada — necesita una ventana de las últimas lecturas del mismo
pozo (idealmente 10 o más), porque dos de las features son medias móviles
que requieren historial. Ver
[docs/adr/004-inference-input-contract.md](docs/adr/004-inference-input-contract.md).

## Estado

🚧 Work in progress — M0-M10 completos (scaffolding, datos, features,
baselines, explicabilidad, registry + inferencia, API FastAPI, frontend
Streamlit, Docker, CI). Deploy real a AWS ECS Fargate sin arrancar. Ver
[docs/adr/](docs/adr/) para las decisiones de diseño tomadas hasta el
momento.

## Cómo correrlo localmente

### Con Docker (backend + frontend juntos)

```bash
docker compose up --build
# backend: http://localhost:8010  (ver docker-compose.yml para el mapeo de puertos)
# frontend: http://localhost:8511
```

El modelo va empaquetado dentro de la imagen del backend en build time, no se
carga desde MLflow en runtime — ver
[docs/adr/006-model-packaging-deploy.md](docs/adr/006-model-packaging-deploy.md).

### Sin Docker

```bash
uv pip install -e ".[ml,api,frontend,dev]"

# Backend (API):
uvicorn backend.app.main:app --reload

# Frontend (en otra terminal, con el backend ya corriendo):
streamlit run frontend/streamlit_app/app.py
```

La predicción nunca calcula SHAP por defecto (`/predict`); pedir la
explicación es una llamada aparte (`/explain`), más cara — ver
[docs/adr/005-shap-endpoint-design.md](docs/adr/005-shap-endpoint-design.md).
Ambos endpoints, y el dashboard de Streamlit, señalan explícitamente cuándo
una predicción es menos confiable (ventana corta, o profundidad dentro de
la zona de alto error ya documentada) en vez de devolverla sin avisar.

## Resultados y límites conocidos

Este proyecto no documenta solo un modelo — documenta los límites reales
de aplicar ML a un dataset de 7 pozos, con diagnóstico reproducible de
por qué. El LightGBM tuneado de M4 no logra superar a un baseline
trivial (media global) en el test final, pese a mostrar una ventaja
clara en cross-validation; cuatro rondas de diagnóstico en
[docs/m4_results.md](docs/m4_results.md) muestran que la causa raíz es
la escasez de pozos por régimen geológico, no la complejidad del modelo
— incluyendo un caso de manual de un clasificador que aprendió una
coincidencia de la muestra de entrenamiento en vez de la señal real.

## Arquitectura

- `ml/` — features, entrenamiento, inferencia, evaluación y
  explicabilidad. Paquete independiente del backend.
- `backend/` — API (FastAPI) que sirve el modelo entrenado.
- `frontend/` — interfaz Streamlit (MVP).
- `docs/adr/` — Architecture Decision Records del proyecto.

## Licenciamiento

Este proyecto distingue explícitamente la licencia del **código** de la
licencia de los **datos** utilizados, porque son distintas.

### Código

El código de este repositorio se distribuye bajo licencia **MIT**. Ver
[LICENSE](LICENSE).

### Datos

Este proyecto utiliza el dataset **USROP** (University of Stavanger Rate
of Penetration), distribuido bajo licencia **CC BY-NC-SA 4.0** (uso no
comercial). Como consecuencia:

- Los archivos CSV del dataset **no se versionan** en este repositorio
  (ver `.gitignore`).
- Se provee un script de descarga documentado con verificación por
  checksum en lugar de distribuir los datos crudos.
- Cualquier modelo o predicción derivados de estos datos hereda la
  restricción de uso no comercial.

Ver [docs/adr/001-dataset-selection.md](docs/adr/001-dataset-selection.md)
para el detalle completo de la decisión.

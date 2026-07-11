# DrillPilot

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

🚧 Work in progress — M0-M6 completos (scaffolding, datos, features,
baselines, explicabilidad, registry + inferencia). M7 (API) sin arrancar.
Ver [docs/adr/](docs/adr/) para las decisiones de diseño tomadas hasta el
momento.

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

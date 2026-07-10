# DrillPilot

Plataforma open source de Machine Learning para predecir el Rate of
Penetration (ROP) durante la perforación de pozos de petróleo, construida
con arquitectura de producción (no un notebook) y foco en explicabilidad
(SHAP) sobre cada predicción.

## Estado

🚧 Work in progress — en fase de scaffolding (M0). Ver
[docs/adr/](docs/adr/) para las decisiones de diseño tomadas hasta el
momento.

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

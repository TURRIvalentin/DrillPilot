# M8: Frontend Streamlit

## Qué se construyó

```
frontend/streamlit_app/
  api_client.py    Cliente HTTP puro (predict/explain/health) -- sin dependencia de
                    Streamlit, testeable sin la app corriendo. BackendError expone
                    el detail que manda backend/app/core/exceptions.py.
  formatting.py     confidence_warnings(): la unica fuente de los 2 mensajes de
                    advertencia (known_limitation_zone, insufficient_history).
  sample_data.py    Carga una ventana real de USROP (ml.features.dataset) para
                    poblar el formulario -- no toca features ni el modelo.
  app.py           La pagina: carga de ejemplo + editor de la ventana, botones
                    Predecir / Explicar (SHAP), resultados con ambos flags
                    visibles como st.warning (no solo JSON) y grafico de barras
                    de contribuciones SHAP.
```

`app.py` nunca llama a `ml.inference` ni carga el modelo -- solo habla HTTP con el
backend de M7 (`api_client.predict`/`explain`/`health`), consistente con ADR-004
(servicio stateless, cliente manda la ventana completa) y ADR-005 (predict/explain
separados por costo de SHAP).

## Los dos flags, visibles de un vistazo

`_render_predict_results` agrega una columna "Zona de alto error" (⚠️ si / no) por
fila y muestra un `st.warning` arriba de la tabla si **alguna** predicción cae en
`known_limitation_zone`, o si `insufficient_history` está activo para toda la
ventana. `_render_explain_results` hace lo mismo para la lectura explicada. Ambos
casos usan `formatting.confidence_warnings()` -- un solo lugar para el texto de
advertencia, no duplicado entre las dos vistas.

Los mensajes citan la fuente del hallazgo (`docs/m6_results.md` para la zona de
profundidad, `docs/adr/004-inference-input-contract.md` para el historial mínimo),
igual que ya hacían los `description` de los schemas del backend.

## Prueba manual end-to-end (backend + frontend reales, sin mocks)

Con `uvicorn backend.app.main:app --port 8000` y
`streamlit run frontend/streamlit_app/app.py` corriendo juntos (el frontend
apuntado al backend real vía `DRILLPILOT_BACKEND_URL`), se ejecutó el script real de
la app con `streamlit.testing.v1.AppTest` -- sin mockear nada, ni el dataset ni las
llamadas HTTP -- para verificar el flujo completo tal como lo vería un usuario:

1. Carga inicial: ventana de ejemplo real (pozo 3, 10 filas, MD 1306.5-1309.3 m).
2. Click en "Predecir" contra el backend real -- devolvió las mismas 10
   predicciones ya verificadas manualmente en M7 (ej. MD=1306.525 ->
   ROP=10.069566), columna "Zona de alto error"="no" en todas, sin warnings.
3. Cambiando a pozo 0, fila inicial 3066 (el primer índice real con MD dentro de
   634-988, calculado directamente sobre el dataset, no adivinado) con la ventana
   completa (10 filas): apareció el warning de `known_limitation_zone` y la tabla
   mostró "⚠️ si" en las 3 primeras filas verificadas (MD=634.011 ->
   ROP=38.299719, igual al valor ya confirmado en M7).
4. Misma posición pero con ventana de 3 filas: aparecieron **ambos** warnings
   simultáneamente (`known_limitation_zone` e `insufficient_history`), confirmando
   que se combinan correctamente cuando las dos condiciones se cumplen a la vez.

Además, correr la app real expuso un warning de deprecación real de Streamlit
1.59 (`use_container_width` -- ver más abajo), no visible al mirar solo el código.

## Tests

`tests/frontend/` (20 tests, todos pasando):

- `test_api_client.py` (5): request/response shape correcto para predict/explain,
  `BackendError` con el `detail` del backend en errores no-2xx, fallback a texto
  crudo si el body no es JSON, `health()` propaga `HTTPError`.
- `test_formatting.py` (4): las 4 combinaciones de los 2 flags (ninguno, cada uno
  solo, ambos).
- `test_sample_data.py` (5): selección de pozo/slice correcta, orden ascendente
  por MD, coerción de tipos (`well_id`→int, `gr_imputed`→bool), solo las columnas
  de `Reading`, round-trip de una tabla editada.
- `test_app.py` (6, vía `streamlit.testing.v1.AppTest` -- corre el script real, no
  solo funciones sueltas): la app carga sin excepción y con una ventana de
  ejemplo poblada; cambiar de pozo reemplaza la ventana; `/predict` muestra ambos
  warnings cuando el mock de `api_client.predict` los activa, y ninguno cuando no;
  un `BackendError` de `/predict` se muestra con `st.error`; `/explain` muestra el
  warning de `known_limitation_zone` y el `st.metric` con la predicción.

Fixture compartida (`tests/frontend/conftest.py`): `patched_dataset` monkeypatchea
`ml.features.dataset.load_combined_dataset` con un frame sintético de 7 pozos x 15
filas (mismo patrón de dict-por-fila que `tests/backend/conftest.py`) y limpia
`st.cache_data` antes de cada test -- ningún test depende de que el CSV real de
USROP esté descargado localmente, consistente con cómo
`tests/ml/test_features.py` evita el dataset real también.

Suite completa del proyecto: **105/105 tests pasando** (61 `tests/ml/` + 24
`tests/backend/` + 20 `tests/frontend/`). 9/9 hooks de pre-commit pasando.

## Decisiones y hallazgos durante la implementación

- **`streamlit`/`requests`** en un nuevo grupo `frontend` de `pyproject.toml`;
  `types-requests` agregado a `dev` para que mypy vea el tipado real de
  `requests` en el venv del proyecto. `frontend*` agregado a
  `[tool.setuptools.packages.find]` (faltaba `frontend/__init__.py`, que sí
  existía en `backend/` -- sin él, mypy fallaba con "Source file found twice
  under different module names" al no poder resolver el paquete).
- **`streamlit.testing.v1.AppTest`** (disponible desde streamlit 1.59, la versión
  instalada) permite correr el script real de la app en tests automatizados en
  vez de limitarse a las funciones puras -- usado para `test_app.py`, análogo en
  espíritu a usar el `TestClient` de FastAPI en `tests/backend/`.
- **Por qué `_cached_dataset()` importa `load_combined_dataset` adentro de la
  función, no arriba del archivo**: además de ser una carga perezosa, esto es lo
  que hace que `monkeypatch.setattr("ml.features.dataset.load_combined_dataset",
  ...)` funcione en los tests -- el import se resuelve en el momento de la
  llamada contra el módulo real (ya en `sys.modules`), no queda fijado al cargar
  `app.py`.
- **Timeout de `AppTest.run()`**: la primera corrida de `AppTest` en un proceso
  paga un costo de arranque (runtime de Streamlit + inicialización de la cache)
  que supera el timeout por defecto de 3s; subieron a 20s en los tests para
  cubrir ese arranque en frío sin quedar dependiente del orden de ejecución de
  los tests.
- **`use_container_width` deprecado**: al correr la app real (no solo los tests)
  contra streamlit 1.59.1 apareció un warning de deprecación
  (`use_container_width` será removido; reemplazo: `width="stretch"`/`"content"`)
  que ningún test hubiera mostrado por si solo (no rompía nada, solo lograba
  warning en consola) -- se corrigió en `app.py` antes de cerrar la milestone.
- **Ventana de ejemplo por defecto**: pozo 3, 10 filas -- fuera de la zona de
  limitación conocida y exactamente en el mínimo recomendado por ADR-004, para
  que la carga inicial de la página no muestre warnings por defecto (el usuario
  ve las advertencias solo cuando realmente aplican, no como ruido constante).

## Fuera de alcance (explícitamente, por instrucción del usuario)

Docker y CI (M9-M10) quedan para la próxima etapa; no se tocó nada de eso en esta
milestone.

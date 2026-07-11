# M6 — Registry promotion + ml/inference

## Promoción en el registry de MLflow

`python -m ml.training.promote_model` (`ml/training/promote_model.py`) empaqueta el
candidato confirmado de M5 (LightGBM de 14 features, run `7a98edb3...`) junto con
`USROPFeatureTransformer` en un único `sklearn.Pipeline` — un solo artefacto,
features + modelo, sin reimplementar nada en el backend, como se fijó en el diseño
original del proyecto.

- Modelo registrado: `drillpilot-rop`, versión **1**.
- Alias: **`production`** (`models:/drillpilot-rop@production`).
- Tag `source_candidate_run_id` en la versión registrada → `7a98edb3...`, trazable
  hasta el candidato de M5.

**Nota de terminología:** el registry de MLflow por *Stages* (`Staging`/`Production`
como estado del modelo) está deprecado desde MLflow 2.9+ a favor de *aliases* + tags.
Este proyecto corre MLflow 3.14 — se usa `set_registered_model_alias(..., "production", ...)`,
el mecanismo vigente para "esta es la versión que debe servirse", no el `stage`
literal que pedías. Mismo objetivo, API no deprecada.

## `ml/inference/predict.py`

Carga el artefacto combinado y expone `predict_rop(history, model=None)`. Diseño:

- **Input: un historial por pozo/sesión (`DataFrame` ordenado por MD ascendente),
  no una fila aislada.** Las features de ventana (`WOB_rolling_mean`,
  `RPM_rolling_mean`) necesitan filas anteriores del mismo pozo para calcularse
  correctamente — pedir una predicción de una sola fila sin historial no tiene una
  respuesta correcta posible dado cómo está construido el modelo. Esto es una
  decisión de diseño real, no un detalle: cualquier API futura (M7) va a tener que
  mantener/recibir ese historial, no solo la lectura instantánea.
- **`well_id` es puramente una clave de agrupación interna**, no necesita coincidir
  con ninguno de los 7 pozos de entrenamiento — nunca se usa como feature del
  modelo (ver `docs/feature_dictionary.md`). Cualquier identificador consistente de
  "esta sesión" sirve.
- **No se reimplementa ninguna lógica de features acá.** `predict_rop` no valida
  columnas ni calcula nada — delega enteramente en el pipeline cargado
  (`USROPFeatureTransformer` dentro del `Pipeline`), así que no puede desalinearse
  de M3 con el tiempo.
- `model` es un parámetro opcional: pasarlo explícitamente (cargado una vez al
  arrancar un futuro servicio) evita ida y vuelta al registry en cada llamada.

4 tests en `tests/ml/test_predict.py` (offline, sin tocar MLflow: usan un
`Pipeline` de prueba con `GlobalMeanBaseline` en vez de LightGBM) — cubren forma
de la salida, que no se toca el registry cuando se pasa el modelo, validación de
columnas delegada, y que `well_id` no necesita ser uno de los 7 pozos de
entrenamiento.

## Validación end-to-end contra el modelo real

Se cargó el modelo recién promovido (`load_production_model()`) y se predijo sobre
datos reales del pozo 0 de test. Las predicciones **coinciden exactamente** (hasta
el float) con las del camino de evaluación oficial (`transformer.transform` +
`bare_model.predict` por separado) — confirma que empaquetar todo junto en un
`Pipeline` no introduce ninguna diferencia de comportamiento.

## Hallazgo no buscado, encontrado validando la inferencia: el error varía fuerte según la profundidad dentro de un mismo pozo

Al probar `predict_rop` sobre el pozo 0 completo (13,746 filas, no solo un
sample), el error resultó muy lejos de ser uniforme dentro del propio pozo:

| Tramo (filas desde el inicio del pozo) | MD hasta | MAE |
|---|---|---|
| Primeras 20 | 496 m | 24.69 |
| Primeras 100 | 506 m | **28.57** (peor tramo) |
| Primeras 500 | 524 m | 15.22 |
| Primeras 2,000 | 585 m | **9.44** (mejor tramo) |
| Primeras 5,000 | 727 m | 11.88 |
| Todo el pozo (13,746) | 1,206 m | 17.47 (el número ya reportado en M5) |
| Últimas 2,000 | desde 1,114 m | **31.02** (peor que el arranque) |

El error es alto al arranque del pozo (MD~490-510m, antes de que las features de
ventana tengan historial suficiente), baja a su mínimo alrededor de MD~585-730m, y
**vuelve a dispararse en el tramo final (MD≥1,114m) — peor incluso que el
arranque.** Esto no es ruido aleatorio: MD≥1,114m está adentro (o pasando) del
gap de profundidad (634-988m) sin cobertura de ningún pozo atípico del CV-pool ya
identificado en M4 (experimento de régimen) y M5 (dependence plot de SHAP) — es la
**tercera** confirmación independiente del mismo fenómeno, esta vez en las
predicciones reales de producción, no en un diagnóstico auxiliar (el router de
régimen de M4) ni en una explicación post-hoc (SHAP de M5).

**Implicancia práctica, no resuelta acá:** el MAE de 17.47 reportado para el pozo 0
es un promedio que esconde un rango real de ~9 a ~31 según en qué profundidad del
pozo se esté parado. Cualquier consumidor de este modelo (M7/FastAPI) tiene que
saber que la confiabilidad de la predicción no es uniforme dentro de un mismo
pozo — es peor precisamente en los dos extremos: el arranque (pocas filas de
historial) y las profundidades fuera del rango que el CV-pool cubrió para el
régimen atípico. Queda como limitación operacional documentada, en la misma línea
de honestidad que el resto del proyecto — no se intenta resolver ni ocultar acá.

## Reproducibilidad

- `python -m ml.training.promote_model` reproduce la promoción (usa el run tageado
  `m5_candidate=true` más reciente).
- `ml.inference.predict.load_production_model()` carga `models:/drillpilot-rop@production`.
- Artefacto local en `mlruns/` (gitignored, no commiteado) — igual que el resto del
  tracking de este proyecto.

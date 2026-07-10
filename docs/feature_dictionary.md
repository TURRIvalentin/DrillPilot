# Diccionario de features (M3)

Producido por [`ml.features.pipeline.USROPFeatureTransformer`](../ml/features/pipeline.py),
que recibe el output de `ml.features.dataset.load_combined_dataset` (datos limpios de
M2 + columna `well_id`) y devuelve la matriz de 16 features documentada acá. No incluye
selección de modelo, tuning ni entrenamiento — eso es M4.

## Columnas de entrada excluidas de la matriz de features (y por qué)

| Columna | Por qué no es una feature |
|---|---|
| `ROP` | Es el target (`y`), no un input. Incluirla sería leakage directo. |
| `well_id` | Bookkeeping para agrupar ventanas temporales y para el split de ADR-003. Usarla como feature dejaría que el modelo memorice "el pozo X tiende a tener ROP Y" — inútil (y engañoso) para un pozo nuevo en inferencia real, que es exactamente el escenario que el split de ADR-003 evalúa. |

**Considerado y descartado para esta iteración:** una feature autoregresiva de ROP
(ej. media móvil de ROP de las últimas N filas). No es leakage per se si se calcula
solo hacia atrás (el driller conoce su ROP reciente en tiempo real), pero se dejó
fuera de esta primera versión para no mezclar la ambigüedad de "feature derivada del
propio target" con el resto de las features directas — candidato a evaluar en una
iteración posterior, no en M3.

## Features directas (12)

Las 11 columnas predictoras de USROP (de las 12 originales, sin `ROP`) más
`gr_imputed`, agregada en M2. Ver [docs/data_dictionary.md](data_dictionary.md) para
unidad y rango completo de cada una — no se repite acá.

| Feature | Justificación como input del modelo |
|---|---|
| `MD` | Profundidad medida — parámetro operacional siempre conocido durante la perforación (variable esperada del proyecto original, "Bit Depth"). |
| `WOB` | Weight on Bit — parámetro de control directo del driller, variable esperada del proyecto original. |
| `SPP` | Presión de standpipe — refleja resistencia hidráulica/estado del sistema de circulación en tiempo real. |
| `T` | Torque en superficie — variable esperada del proyecto original, relacionada con la interacción broca-formación. |
| `RPM` | Velocidad rotatoria — parámetro de control directo, variable esperada del proyecto original. |
| `FR` | Caudal de lodo — afecta limpieza del pozo y transporte de ripios, disponible en tiempo real. |
| `DS` | Densidad de lodo — parámetro de control, disponible en tiempo real. |
| `HD` | Diámetro de hoyo/broca — cambia por corrida de broca, disponible siempre que se conozca la sarta actual. |
| `HL` | Hookload — variable esperada del proyecto original. |
| `VD` | Profundidad vertical verdadera — complementa a MD en pozos direccionales. |
| `GR` | Gamma ray — proxy de litología, disponible en tiempo real (post-imputación de M2 donde corresponde). |
| `gr_imputed` (0/1) | Trazabilidad de la imputación de GR (regla 3, M2): deja que el modelo (y SHAP) distingan explícitamente una lectura medida de una interpolada, en vez de tratarlas como igualmente confiables. |

## Features de ventana (4)

Todas calculadas con `groupby("well_id")` **antes** de cualquier `.rolling()`/`.diff()`
(nunca sobre el DataFrame concatenado directo) y **solo hacia atrás** — sin
`center=True`, sin `shift` negativo — per
[ADR-003 §3](adr/003-split-strategy.md). `tests/ml/test_features.py` verifica ambas
propiedades de forma explícita: no cruzan el límite entre pozos
(`test_no_leakage_across_well_boundary`) y no miran hacia adelante
(`test_no_lookahead_truncating_future_rows_does_not_change_past_features`).

Ventana por defecto: `N=10` filas anteriores dentro del mismo pozo (parámetro
`rolling_window` del transformer, no hardcodeado).

| Feature | Fórmula | Justificación física |
|---|---|---|
| `WOB_rolling_mean_{N}` | Media móvil de WOB, últimas N filas del mismo pozo | Proxy de la tendencia reciente de carga sobre la broca — distingue una carga sostenida de un pico transitorio, más informativo que el valor instantáneo solo. |
| `RPM_rolling_mean_{N}` | Media móvil de RPM, últimas N filas del mismo pozo | Proxy del régimen reciente de rotación — ayuda a distinguir tramos sostenidos de perforación rotativa de tramos de deslizamiento (ver `docs/eda_findings.md`, sección RPM==0), más allá del valor puntual de una sola fila. |
| `T_rolling_std_{N}` | Desvío estándar móvil de Torque, últimas N filas del mismo pozo | Proxy de perforación errática/vibración reciente (stick-slip u otra disfunción) — un torque inestable en la ventana reciente es información que el valor instantáneo de torque no captura. |
| `WOB_diff_1` | `WOB[fila actual] − WOB[fila anterior]` dentro del mismo pozo | Proxy de qué tan agresivamente está cambiando el driller el peso sobre la broca ahora mismo — la tasa de cambio, no el nivel, que tiene su propia relación con la respuesta transitoria del ROP. |

### Manejo del primer registro de cada pozo (borde de ventana)

- `*_rolling_mean_{N}` usa `min_periods=1`: la primera fila de cada pozo devuelve su
  propio valor (media de 1 elemento), nunca `NaN`.
- `T_rolling_std_{N}` con una sola observación no tiene desvío definido
  (`std` de 1 valor es indefinido en pandas) — se completa con `0.0` explícitamente
  ("sin variabilidad observada todavía"), no se deja `NaN`.
- `WOB_diff_1` no tiene fila anterior en la primera posición de cada pozo — se
  completa con `0.0` explícitamente ("sin cambio registrado todavía"), no se deja
  `NaN`.

Resultado: la matriz de features no tiene ningún `NaN` (verificado contra los datos
reales de los 7 pozos, 198,928 filas, 0 valores faltantes en la matriz final).

## Nota sobre leakage entre train/CV-pool y test

El transformer no aprende ni guarda ninguna estadística de los datos con los que se
ajusta (`fit`) — cada feature se calcula fila por fila, usando solo el historial de
esa fila dentro de su propio pozo. No hay parámetros aprendidos en train (ej. medias
globales) que se reutilicen sobre CV-pool o test, a diferencia de un `StandardScaler`
u otro transformer con estado. Esto se corresponde con la decisión de ADR-002 de usar
LightGBM (no requiere escalado) y elimina una categoría entera de leakage que sí
existiría si se introdujera normalización más adelante sin cuidado.

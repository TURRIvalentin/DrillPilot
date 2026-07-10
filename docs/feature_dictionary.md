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

### Manejo del primer registro de cada pozo (borde de ventana) — qué significa cada valor

Hay **dos estrategias distintas** en las 4 features, no una sola convención genérica,
y ninguna de las dos produce un valor estadísticamente equivalente al de una fila con
historial completo. Se documentan por separado para no dar a entender que "no hay
`NaN`" es lo mismo que "todas las filas son igual de confiables".

**`WOB_rolling_mean_{N}` y `RPM_rolling_mean_{N}` — ventana parcial (`min_periods=1`).**
No hay relleno artificial: la fila *k* de cada pozo (con *k* = 0, 1, ..., N-2) calcula
la media sobre las *k+1* muestras disponibles hasta ese punto, no sobre *N*. Esto
implica una escala continua de confiabilidad, no un corte binario:

- Fila 0 de cada pozo: la "media móvil" es literalmente el valor de esa única fila —
  **cero suavizado**, no distinta de usar la columna directa sin ventana.
- Fila 1: media de 2 muestras — todavía muy sensible a una sola muestra ruidosa.
- ...
- Fila N-1 en adelante: primera media sobre la ventana completa de *N* muestras — a
  partir de acá el valor sí es comparable entre filas del mismo pozo.

En la práctica, con `N=10` (default), las primeras 9 filas de cada uno de los 7 pozos
(63 filas de 198,928, 0.03% del dataset) tienen una media calculada sobre menos
muestras de las nominales. No se excluyen ni se marcan con un flag separado en esta
iteración — a diferencia de `gr_imputed`, que sí distingue explícitamente valor medido
de imputado, acá esa distinción no existe todavía. Queda anotado como una asimetría
menor entre features, no resuelta en M3.

**`T_rolling_std_{N}` — ventana parcial + caso degenerado en la fila 0.** Comparte el
comportamiento de ventana parcial de arriba (filas 1 a N-2 con menos de N muestras),
pero la fila 0 es un caso distinto, no solo "una muestra": el desvío estándar de una
sola observación no está matemáticamente definido (pandas devuelve `NaN`, no `0`), así
que el `0.0` que aparece ahí **no es una medición de "sin variabilidad" — es un
placeholder elegido para no dejar `NaN` en la matriz**, sin contenido estadístico
propio. A partir de la fila 1 (2+ muestras) el desvío ya es un número real, aunque de
baja precisión hasta llegar a la ventana completa en la fila N-1.

**`WOB_diff_1` — sin ventana parcial, `fillna(0.0)` puro.** No es un caso de "menos
muestras de las nominales": en la fila 0 de cada pozo no existe *ninguna* fila
anterior dentro de ese pozo, así que el `diff` no tiene con qué calcularse (a
diferencia de las medias/desvíos móviles, acá no hay una versión "parcial" posible con
0 muestras previas). El `0.0` es una **asunción de modelado explícita** ("sin cambio
registrado, tratar como si viniera de un estado estable"), no una medición ni una
aproximación con menos datos — es cualitativamente distinta de los dos casos
anteriores y no debería leerse como "el WOB no cambió al empezar a perforar este
tramo", que no es algo que el dataset permita afirmar.

Resultado combinado: la matriz de features no tiene ningún `NaN` (verificado contra
los datos reales de los 7 pozos, 198,928 filas, 0 valores faltantes en la matriz
final), pero "sin `NaN`" no es lo mismo que "sin filas de confiabilidad reducida" — las
primeras `N-1` filas de cada pozo (multiplicado por 7 pozos) llevan features de
ventana menos informativas que el resto, por diseño, no por error.

## Tamaño de ventana (N) en unidades físicas reales

`N=10` filas **no es una unidad física fija** — USROP no tiene una columna de tiempo,
y el espaciado entre filas en profundidad (MD) varía fuerte tanto entre pozos como
dentro de un mismo pozo (ver la anomalía de salto de MD ya documentada en
[docs/cleaning_rules.md](cleaning_rules.md)). Se calculó, sobre los datos reales, tanto
el espaciado en metros como una estimación de tiempo implícito entre filas
(`Δt = ΔMD / (ROP en m/s)`, reproducible a partir de columnas ya existentes — USROP no
loguea timestamps, así que esto es una estimación derivada, no un dato medido).

| Pozo | ΔMD mediana (m) | ΔMD p10–p90 (m) | Δt implícito mediana (s) | Δt implícito p10–p90 (s) |
|---|---|---|---|---|
| 0 | 0.042 | 0.010–0.073 | 4.48 | 1.02–6.65 |
| 1 | 0.051 | 0.012–0.088 | 3.56 | 0.74–5.41 |
| 2 | 0.037 | 0.012–0.107 | 5.58 | 1.99–18.13 |
| 3 | 0.027 | 0.009–0.067 | 5.07 | 1.71–11.19 |
| 4 | 0.031 | 0.012–0.104 | 7.65 | 2.80–24.07 |
| 5 | 0.031 | 0.007–0.085 | 5.07 | 1.57–15.16 |
| 6 | 0.045 | 0.012–0.091 | 4.34 | 0.98–7.68 |
| **Global** | **0.033** | **0.009–0.089** | **5.19** | **1.71–16.15** |

**Con `N=10` (default), la ventana cubre, en la mediana global: ~0.33 m de profundidad
y ~52 segundos** — pero con variación real de casi un orden de magnitud según el pozo
y el tramo (p10–p90 global: ~0.09–0.89 m, ~17–162 s). No es un número fijo utilizable
para una afirmación tipo "la ventana representa los últimos X metros" sin la
salvedad de la variabilidad.

**Por qué se usó la mediana y no la media:** la media de `ΔMD` es prácticamente
idéntica en los 7 pozos (~0.0520 m) — esto no es una coincidencia física, es una
identidad aritmética (`ΔMD_media = (MD_max − MD_min) / (n_filas − 1)`), y queda
inflada por saltos de MD grandes y raros entre filas consecutivas (el mismo tipo de
anomalía de 25 m ya documentada en `cleaning_rules.md` para el pozo 4 — no es un caso
único, hay saltos similares, aunque menos extremos, en otros pozos). El `Δt` implícito
tiene el mismo problema todavía más amplificado: la media global es 12.85 s pero el
máximo de una sola fila llega a ~294,500 s (~82 h), claramente un artefacto del salto
de MD dividido por un ROP moderado, no un tiempo de perforación real. La mediana y el
rango p10–p90 son la lectura representativa; la media de `Δt` no se reporta en la
tabla por esta razón.

## Nota sobre leakage entre train/CV-pool y test

El transformer no aprende ni guarda ninguna estadística de los datos con los que se
ajusta (`fit`) — cada feature se calcula fila por fila, usando solo el historial de
esa fila dentro de su propio pozo. No hay parámetros aprendidos en train (ej. medias
globales) que se reutilicen sobre CV-pool o test, a diferencia de un `StandardScaler`
u otro transformer con estado. Esto se corresponde con la decisión de ADR-002 de usar
LightGBM (no requiere escalado) y elimina una categoría entera de leakage que sí
existiría si se introdujera normalización más adelante sin cuidado.

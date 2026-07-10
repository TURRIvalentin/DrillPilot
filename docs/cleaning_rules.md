# Reglas de limpieza propuestas (M2)

Basado en [docs/eda_findings.md](eda_findings.md). Cada regla lista su
justificación física/operacional y el estado de aprobación. **Ninguna regla
de acá está implementada todavía** — quedan pendientes de aprobación antes de
escribir el código de limpieza (paso 3 de M2).

## Regla 1 — No eliminar ni transformar filas con RPM == 0

**Acción:** ninguna. Las filas con `RPM == 0` se mantienen exactamente como
están.

**Justificación:** en el 100% de las filas con `RPM == 0` en los 7 pozos, el
WOB está en rango de perforación activa normal y el ROP nunca cae por debajo
de 0.5 m/h — es decir, nunca aparecen junto con la firma de una parada real
(`WOB≈0 & RPM≈0`, que se verificó no ocurre en ninguna fila de todo el
dataset). Es la firma de perforación deslizante ("sliding") con motor de
fondo: el sensor de RPM en superficie marca 0 mientras el trépano sigue
rotando por acción del motor de lodo. Tratar estas filas como error
introduciría un sesgo sistemático: se estaría enseñando al modelo a ignorar
o distorsionar un régimen de perforación real y frecuente en pozos
direccionales.

**Riesgo de no actuar:** ninguno identificado — es la opción sin riesgo,
dado que la evidencia no admite otra interpretación.

**Estado:** ✅ aprobada.

## Regla 2 — No tratar ROP bajo (< 1 m/h) como outlier

**Acción:** ninguna. No se elimina, no se clippea, no se transforma.

**Justificación:** en el 100% de las filas con `ROP < 1.0 m/h`, WOB y RPM
están en rango de perforación activa (`WOB ≥ 0.5`, `RPM ≥ 1`) — son
mediciones reales de avance lento en formación dura, no ruido ni paradas.
Eliminarlas o clippearlas sesgaría el modelo exactamente en el régimen que
más le cuesta predecir bien (el riesgo de ROP≈0 ya identificado en el
análisis inicial del proyecto), y por el cual se fijó MAE como métrica
primaria en vez de un error porcentual.

**Riesgo de no actuar:** ninguno identificado.

**Estado:** ✅ aprobada.

## Regla 3 — Imputar GR == 0 por interpolación lineal dentro de cada pozo

**Acción:** para las filas con `GR == 0` (197 de 198,928, 0.099% del
dataset, presentes solo en los pozos 2, 3, 4 y 5), reemplazar el valor de
`GR` por interpolación lineal usando los valores válidos (no-cero) más
cercanos dentro del mismo pozo, ordenando por el índice original de fila
(que sigue MD creciente). Si el bloque de ceros está al borde del pozo (sin
vecino válido de un lado), usar forward-fill o backward-fill según
corresponda desde el valor válido disponible — el mismo criterio que ya
usaron los autores originales de USROP para otros gaps de logueo (ver
[docs/data_dictionary.md](data_dictionary.md#limitación-conocida-relleno-forwardbackward-aplicado-por-los-autores)),
así que es metodológicamente consistente con cómo se construyó el resto del
dataset. Se agrega una columna booleana `gr_imputed` marcando exactamente
qué filas fueron tocadas — a diferencia del relleno original de Tunkiel et
al., acá la imputación queda trazable en vez de indistinguible del dato
medido.

**No se elimina la fila completa.** Las otras 11 columnas de esas filas
(incluyendo RPM, WOB y ROP) son mediciones válidas de perforación activa —
eliminarlas tiraría datos buenos de todas las demás variables solo por un
problema puntual en una columna.

**Justificación:** el patrón observado (bloques cortos de 1 a 24 filas
contiguas, dispersos en profundidades medias/tardías del pozo, sin overlap
con eventos de RPM==0) es consistente con caídas intermitentes de
telemetría del sensor de gamma ray (mud-pulse/LWD), no con un valor
geológico real — el gamma ray de fondo prácticamente nunca es exactamente 0
durante decenas de muestras consecutivas en secuencias sedimentarias
reales. Se descartó la hipótesis alternativa ("sensor todavía no llegó a
profundidad") porque los rangos de MD en `GR==0` no están al inicio del
pozo.

**Riesgo de no actuar:** dejar `GR==0` sin corregir introduce un valor
físicamente inverosímil que un modelo de árboles podría aprender a usar
como si fuera informativo (ej. asociar `GR==0` a alguna característica
espuria de esos pozos en vez de tratarlo como ausencia de dato), y afecta a
SHAP: la importancia de GR quedaría contaminada por un artefacto de sensor
en 4 de los 7 pozos.

**Riesgo de actuar (a favor de que lo revises):** la interpolación asume
que el gamma ray real varía suavemente en esos tramos cortos (razonable a
esta escala de profundidad — decenas de filas equivalen a pocos metros),
pero es una asunción, no un hecho verificado fila por fila.

### Longitud de los gaps: ¿puntos aislados o corridas consecutivas?

Verificado antes de asumir que interpolación lineal simple alcanza, con la
extensión real en profundidad (MD) que cubre cada corrida, no solo el
conteo de filas:

| Pozo | Corridas (largo en filas) | MD span por corrida |
|---|---|---|
| 2 | 24, 2 | 0.759 m, 0.016 m |
| 3 | 9, 8 | 0.296 m, 0.223 m |
| 4 | 8, 16, 11, **11**, 4, 11, 1, 5, 2, 4, 10, 3, 2, 11, 10, 7 | 0.265, 0.631, 0.808, **25.707**, 0.131, 0.589, 0.000, 0.350, 0.137, 0.128, 0.433, 0.089, 0.079, 0.698, 0.143, 0.140 m |
| 5 | 16, 5, 4, 10, 3 | 1.040, 0.210, 0.107, 0.585, 0.071 m |

Son corridas consecutivas (no puntos aislados dispersos), con hasta 24
filas seguidas — dispara el umbral de aviso de >10 muestras. Sin embargo,
en 195 de las 197 filas afectadas, el span de profundidad real es
**menor a 1.1 m** — a esa escala, perforación prácticamente estacionaria,
interpolación lineal es una asunción segura.

**Excepción encontrada y resuelta con el usuario:** la corrida de 11 filas
en el pozo 4 (índices 19712-19722) cubre **25.7 m** de MD, no fracciones de
metro como el resto — producto de dos saltos de MD grandes entre filas
consecutivas (+13.4 m y +24.6 m) que no ocurren en ningún otro gap. El GR
inmediatamente antes (65.0→73.55→39.8) y después (30.87→34.67→28.9→37.75)
del gap difiere de forma compatible con un cambio real de litología, así
que la interpolación lineal ahí tiene más incertidumbre que en el resto de
los casos. **Decisión (confirmada explícitamente):** se interpola de la
misma forma que el resto, marcada con `gr_imputed=True` igual que las
demás — se prioriza mantener una única regla simple y trazable en vez de
una excepción especial en el código, dejando esta nota como limitación
conocida en vez de una lógica separada no testeable de forma clara.

**Estado:** ✅ aprobada, con la excepción de arriba resuelta.

---

## Fuera de scope de estas reglas (no investigado en M2, no propuesto)

No se investigaron en el EDA de M2 — y por lo tanto no se proponen reglas
sobre — outliers en WOB/Torque/SPP/Hookload por encima de rangos normales,
filas duplicadas, ni transformaciones de escala/normalización. Proponer una
regla de limpieza sin evidencia de EDA que la respalde violaría el mismo
criterio que se aplicó acá (decisión física, no estadística a ciegas). Si
querés que se investigue alguno de estos puntos, es una vuelta adicional
sobre el EDA antes de sumarlo acá.

## Qué pasa después de aprobar

Una vez aprobado este documento, el paso 3 de M2 implementa las reglas 1-3
en `ml/cleaning/` (no en `ml/features/`: limpieza corrige problemas de
calidad de dato ya identificados, feature engineering construye variables
derivadas nuevas — son responsabilidades distintas, y `ml/features/` queda
reservado para el pipeline sklearn-compatible del milestone siguiente).
Regla 1 y 2 quedan como no-ops explícitos y testeados — "esta función no
debe tocar estas filas" es un test tan válido como cualquier otro — y la
regla 3 como la única transformación real, con un test unitario por regla
en `tests/ml/`.

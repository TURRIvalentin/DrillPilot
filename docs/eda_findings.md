# EDA findings (M2) — enfocado en decisiones de limpieza

Generado por [`ml/eda/generate_eda_report.py`](../ml/eda/generate_eda_report.py)
sobre los 7 CSV reales en `data/raw/`. Los números citados acá son
reproducibles corriendo ese script — no están tipeados a mano; salen de
[`docs/eda/zero_value_investigation.json`](eda/zero_value_investigation.json).

No es EDA genérico: el objetivo es decidir qué limpiar y por qué, no explorar
por explorar. Ver también la limitación de forward/backward filling ya
documentada en [docs/data_dictionary.md](data_dictionary.md#limitación-conocida-relleno-forwardbackward-aplicado-por-los-autores).

## 1. Distribución de ROP, WOB, RPM, Torque y SPP por pozo

PNGs en [`docs/eda/`](eda/): `distribution_ROP.png`, `distribution_WOB.png`,
`distribution_RPM.png`, `distribution_T.png`, `distribution_SPP.png`. Cada uno
tiene un histograma por pozo (7 subplots), sin agregar entre pozos.

**Conclusión:** los 7 pozos son claramente heterogéneos entre sí, no solo en
rango sino en forma de la distribución:

- ROP es multimodal en todos los pozos, pero los modos están en lugares
  distintos por pozo (ej. pozo 2 concentra masa entre 0-30 m/h, pozo 1 entre
  30-90 m/h). Esto confirma el riesgo de no-estacionariedad entre pozos ya
  señalado en el análisis original — un modelo global va a tener que lidiar
  con regímenes de ROP muy distintos según el pozo/formación.
- RPM también es multimodal por pozo, con clusters en valores muy distintos
  (ver más abajo, sección 2, para el detalle de los picos en 0).
- No se ve ningún pozo con una distribución degenerada (una sola barra, un
  solo valor constante) en ninguna de las 5 variables — descarta la
  hipótesis de "sensor pegado" a nivel de pozo completo.

No se investigan aquí correlaciones entre variables ni se decide
normalización/transformación — eso es feature engineering (milestone
siguiente), fuera de scope de M2.

## 2. RPM == 0 y GR == 0

### RPM == 0

| Pozo | Filas | RPM==0 | % | WOB medio en RPM==0 | ROP medio en RPM==0 | % ROP<0.5 en RPM==0 | Bloques contiguos |
|---|---|---|---|---|---|---|---|
| 0 | 13,746 | 289 | 2.10% | 18.92 | 31.57 | 0.0% | 1 bloque de 289 |
| 1 | 6,389 | 0 | 0.00% | — | — | — | — |
| 2 | 47,645 | 25 | 0.05% | 3.70 | 13.96 | 0.0% | 1 bloque de 25 |
| 3 | 53,041 | 1 | 0.00% | 1.18 | 17.81 | 0.0% | 1 bloque de 1 |
| 4 | 51,708 | 11 | 0.02% | 22.15 | 51.94 | 0.0% | bloques de 1 y 10 |
| 5 | 18,548 | 0 | 0.00% | — | — | — | — |
| 6 | 7,851 | 0 | 0.00% | — | — | — | — |

**Conclusión: RPM==0 es un evento operacional legítimo de perforación
deslizante ("sliding") con motor de fondo, no una falla de sensor ni una
conexión/parada.**

Razonamiento físico: en perforación direccional con motor de lodo (mud
motor), el trépano rota impulsado por el flujo de lodo a través del motor de
fondo, sin que la sarta de perforación rote en superficie — el sensor de RPM
(medido en superficie/mesa rotativa) legítimamente marca 0 mientras la
perforación sigue activa. Si RPM==0 correspondiera a una conexión, viaje o
parada real, esperaríamos ver **WOB≈0 y ROP≈0 simultáneamente** (sin peso
sobre el trépano, sin avance). Eso no ocurre en ningún pozo: en el 100% de
las filas con RPM==0, ROP<0.5 nunca se da (0.0% en todos los pozos con datos),
y el WOB medio en esas filas (3.7 a 22.1) es del mismo orden que el WOB de
perforación activa normal del pozo. Además, en los pozos 0 y 2 el evento
aparece como un único bloque contiguo largo (289 y 25 filas respectivamente),
consistente con un intervalo de slide sostenido y deliberado, no con un
glitch puntual de sensor.

Los pozos 1, 5 y 6 nunca tienen RPM==0 — perforaron enteramente en modo
rotativo (o el tramo registrado no incluyó slides).

**Implicancia para limpieza: no eliminar ni imputar filas con RPM==0.**

### GR == 0

| Pozo | Filas GR==0 | Bloques contiguos | Rango MD en GR==0 | Rango MD total del pozo | Overlap con RPM==0 |
|---|---|---|---|---|---|
| 0 | 0 | — | n/a | 491.0–1206.0 | — |
| 1 | 0 | — | n/a | 301.2–633.5 | — |
| 2 | 26 | 24, 2 | 2829.9–3378.9 | 987.9–3466.0 | 0 |
| 3 | 17 | 9, 8 | 2878.2–3040.6 | 1306.5–4065.3 | 0 |
| 4 | 116 | 16 bloques de 1 a 16 filas | 2540.8–3592.1 | 1400.5–4090.0 | 0 |
| 5 | 38 | 5 bloques de 3 a 16 filas | 3034.7–3781.3 | 2828.2–3792.2 | 0 |
| 6 | 0 | — | n/a | 225.2–633.5 | — |

**Conclusión: GR==0 es consistente con caídas intermitentes de telemetría del
sensor de gamma ray (LWD/mud-pulse), no un valor físico real ni un evento
operacional.**

Razonamiento: se descartó la hipótesis inicial de "el sensor todavía no llegó
a profundidad" (que predeciría GR==0 concentrado al inicio del pozo, en el
tramo de MD más bajo) — en los 4 pozos afectados, los rangos de MD en GR==0
están en la sección media/tardía del pozo, no al comienzo. Tampoco hay
overlap con los eventos de RPM==0 (0 filas en común en los 4 pozos), así que
no es un artefacto asociado al modo slide. El patrón que sí se sostiene:
bloques cortos y contiguos (entre 1 y 24 filas, con longitud media de 7 a 13)
dispersos en distintas profundidades del pozo — la firma típica de una
pérdida temporal de señal de telemetría de fondo (por ciclado de bombas,
ruido, o desconexión momentánea de la transmisión mud-pulse), no un valor
geológico real: el gamma ray de fondo casi nunca es exactamente 0 en
secuencias sedimentarias reales durante decenas de muestras consecutivas.

Es un fenómeno raro en términos absolutos: 197 filas de 198,928 totales
(0.099%), concentrado en 4 de los 7 pozos.

**Implicancia para limpieza:** candidato a tratarse como dato faltante del
sensor GR e imputarse (no eliminar la fila completa — las otras 11 columnas
de esas filas, incluyendo RPM y ROP, son mediciones válidas de perforación
activa). Detalle de la regla propuesta en
[docs/cleaning_rules.md](cleaning_rules.md).

## 3. ROP ≈ 0

Umbral usado para la investigación: `ROP < 1.0 m/h` (no es un umbral de
limpieza, es solo el corte para esta investigación).

| Pozo | Filas ROP<1.0 | % | "Parado" (WOB<0.5 & RPM<1) | Drilling activo lento (WOB≥0.5 & RPM≥1) |
|---|---|---|---|---|
| 0 | 24 | 0.17% | 0 | 24 (100%) |
| 1 | 0 | 0.00% | — | — |
| 2 | 39 | 0.08% | 0 | 39 (100%) |
| 3 | 9 | 0.02% | 0 | 9 (100%) |
| 4 | 23 | 0.04% | 0 | 23 (100%) |
| 5 | 0 | 0.00% | — | — |
| 6 | 0 | 0.00% | — | — |

**Conclusión: ROP≈0 no corresponde a paradas/conexiones — es perforación
activa real en formación dura o de avance lento.** En el 100% de las filas
con ROP<1.0, WOB y RPM están en rango de perforación activa (WOB≥0.5,
RPM≥1). Se verificó además, sobre las 198,928 filas del dataset completo
(no solo el subconjunto ROP<1.0), que no existe ninguna fila con WOB<0.5 y
RPM<1 simultáneamente.

**Aclaración importante para no sobre-interpretar esto:** que el patrón
"parado/conexión" no aparezca en los datos **no significa que las
conexiones y viajes no ocurran en la operación real** — significa que es
muy probable que Tunkiel et al. (2021) ya hayan excluido esos tramos de
no-perforación al curar el benchmark USROP, dejando solo muestras
on-bottom. Es una distinción sobre el dataset, no sobre la física de la
perforación: no se puede concluir que las conexiones "no existen", solo que
este dataset filtrado no las incluye. Ver también la nota correspondiente en
[docs/data_dictionary.md](data_dictionary.md#limitación-conocida-relleno-forwardbackward-aplicado-por-los-autores).

**Implicancia para limpieza: no tratar ROP bajo como outlier a eliminar ni a
clippear.** Confirma el riesgo ya identificado en el análisis original del
proyecto (ROP≈0 generando errores porcentuales infinitos) — la razón por la
que MAE es la métrica primaria sigue siendo válida, pero el motivo no es
"hay basura cerca de 0", es que esas muestras son reales y hay que
predecirlas bien.

## 4. Resumen para docs/cleaning_rules.md

| Hallazgo | Acción propuesta |
|---|---|
| RPM==0 (slide con motor de fondo) | Ninguna — mantener sin cambios |
| ROP≈0 (drilling activo lento) | Ninguna — mantener sin cambios, no es outlier |
| GR==0 (caída de telemetría) | Imputar por interpolación dentro del pozo (no eliminar fila) |

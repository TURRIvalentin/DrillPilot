# M4 — Resultados: dummy vs. Bourgoyne & Young reducido (4/8 términos) vs. LightGBM

Números generados por [`ml/training/train_baselines.py`](../ml/training/train_baselines.py)
(`python -m ml.training.train_baselines`), trackeados en MLflow (experimento
`drillpilot-m4-baselines`, local en `mlruns/`, no commiteado) y guardados en
[`docs/m4_metrics.json`](m4_metrics.json) — esta tabla no está tipeada a mano.

## Hallazgo principal (antes que nada, no al final)

**LightGBM tuneado no le gana al dummy (media global) en el test final de ADR-003, y
pierde contra los tres modelos en cada uno de los 3 pozos de test individualmente.**
No es un bug — está verificado contra un dummy evaluado bajo el mismo protocolo
LOWO-CV (ver sección de diagnóstico) — es un hallazgo real sobre qué tan lejos
generaliza un modelo tuneado con solo 4 pozos de entrenamiento a 3 pozos
completamente nuevos.

| Pozo | Régimen | Dummy | B&Y reducido | LightGBM | Mejor modelo |
|---|---|---|---|---|---|
| 0 | atípico | 16.64 | **10.32** | 16.83 | B&Y reducido |
| 3 | dominante | **8.18** | 10.64 | 9.29 | Dummy |
| 5 | dominante | **13.82** | 16.45 | 14.31 | Dummy |

LightGBM no gana en ningún pozo de test individual. Esto contrasta fuerte con su
performance en CV (ver más abajo), donde le gana al dummy por un margen enorme —
la ventaja simplemente no se transfiere a estos 3 pozos nunca vistos.

## Setup de evaluación

Split y CV según [ADR-003](adr/003-split-strategy.md): CV-pool = pozos 1, 2, 4, 6
(113,593 filas); test final = pozos 0, 3, 5 (85,335 filas), nunca tocado hasta esta
evaluación. Tuning (B&Y: umbral de WOB; LightGBM: hiperparámetros vía Optuna,
15 trials) por leave-one-well-out CV sobre el CV-pool, optimizando MAE.
Bourgoyne & Young identificado en todo lugar como **"Bourgoyne & Young reducido
(4/8 términos)"**, nunca sin calificar, per [ADR-002](adr/002-modelo-baseline.md).

## Tabla comparativa completa

| Métrica | Dummy | B&Y reducido (4/8 términos) | LightGBM |
|---|---|---|---|
| CV MAE (LOWO, 4 folds) | 21.52 | 14.07 | **7.79** |
| Test MAE pooled | **10.77** | 11.85 | 11.59 |
| Test MAE — régimen dominante (pozos 3+5) | **9.64** | 12.14 | 10.59 |
| Test MAE — régimen atípico (pozo 0) | 16.64 | **10.32** | 16.83 |
| Test MAE — pozo 0 | 16.64 | **10.32** | 16.83 |
| Test MAE — pozo 3 | **8.18** | 10.64 | 9.29 |
| Test MAE — pozo 5 | **13.82** | 16.45 | 14.31 |

**En CV, el orden es exactamente el esperado** (LightGBM << B&Y << dummy) — la
ventaja del modelo de ML es real y grande dentro de la distribución de
entrenamiento (63.8% mejor que el dummy, mismo protocolo LOWO-CV). **En test, el
orden se invierte casi por completo.**

## Mejora relativa de LightGBM (según lo pedido, no solo pooled)

| Comparación | Pooled | Régimen dominante | Régimen atípico |
|---|---|---|---|
| LightGBM vs. dummy | **-7.7%** (peor) | -9.8% (peor) | -1.1% (peor, prácticamente empatado) |
| LightGBM vs. B&Y reducido | +2.2% (mejor) | +12.8% (mejor) | **-63.0%** (mucho peor) |

Signo negativo = LightGBM tiene *mayor* error (peor) que el comparado. LightGBM le
gana a B&Y reducido en el régimen dominante, pero pierde feo contra B&Y reducido en
el régimen atípico — el único pozo de test que no comparte régimen con ningún pozo
de CV-pool grande.

## Por qué (hipótesis, no un hecho verificado — dejarlo así de honesto)

No hay forma de confirmar esto sin más experimentación (fuera de scope de M4), pero
las hipótesis más consistentes con la evidencia:

1. **LOWO-CV protege contra overfitting a *un* pozo, no contra overfitting al
   *conjunto* de 4 pozos del CV-pool.** Optuna optimiza hiperparámetros que
   funcionan bien rotando entre esos 4 pozos específicos — pero "funciona bien
   entre estos 4" no es lo mismo que "generaliza a un pozo 5 nunca visto". Con
   solo 4 pozos de entrenamiento, esa distinción importa mucho.
2. **La heterogeneidad entre pozos, ya documentada desde M2/ADR-003, es
   probablemente demasiado grande para que un modelo flexible generalice bien con
   tan pocos pozos de referencia.** El dummy no tiene forma de "sobreajustar" —
   su única apuesta es la media, que resulta ser una apuesta robusta cuando cada
   pozo nuevo es, en algún sentido, impredecible desde los 4 que se vieron.
3. **B&Y reducido gana específicamente en el pozo atípico** posiblemente porque su
   forma funcional extremadamente restringida (3 términos físicos) tiene mucha
   menos capacidad de ajustarse a particularidades del CV-pool que no generalizan
   — "menos flexible" se vuelve una ventaja, no una limitación, cuando el pozo de
   test es genuinamente distinto a todo lo visto en entrenamiento.

## Diagnóstico de Bourgoyne & Young reducido

- `wob_threshold` elegido por LOWO-CV: **0.5** (de la grilla `[0.0, 0.1, 0.25, 0.5, 1.0]`,
  unidades de 1000 lbf/in) — no un valor de literatura, ver ADR-002.
- Fracción de filas con `x5` clippeado (W/db por debajo del umbral elegido): **20.5%**
  del CV-pool — una porción no trivial del término queda en el piso numérico, no en
  un valor con contenido físico real. Documentado, no escondido.
- Fracción de filas con `x6` clippeado (RPM≈0, deslizamiento con motor de fondo):
  **0.03%** del CV-pool — bajo, como se esperaba dado que el M2 ya había establecido
  que estos eventos son raros (ver `docs/eda_findings.md`).
- Coeficientes fiteados: `a1=2.602`, `a2=0.00012`, `a5=0.0122`, `a6=0.1625` (unidades
  ya convertidas, ver `ml/models/byoung_reduced.py`).

## Reproducibilidad

- `python -m ml.training.train_baselines` reproduce todo (mismo `random_state=42`
  en B&Y, LightGBM y Optuna's sampler).
- Artefactos en MLflow: params, métricas (pooled + por pozo + por régimen + CV) y el
  modelo serializado de cada corrida, bajo el experimento `drillpilot-m4-baselines`.
- `docs/m4_metrics.json`: la misma información en JSON, fuente de esta tabla.

## Recomendación antes de seguir a M5

Este hallazgo (LightGBM no le gana al dummy en test) es demasiado importante para
avanzar a SHAP/explicabilidad como si no hubiera pasado. Antes de M5, decidir
explícitamente uno de estos caminos (no se eligió ninguno acá, es una decisión de
diseño):

1. **Seguir a M5 igual, documentando esto como limitación conocida del MVP** —
   SHAP sobre un modelo que no supera al dummy en test sigue siendo válido para
   explicar *qué aprendió* el modelo, aunque no sea la mejor noticia sobre si
   *debería* reemplazar al dummy en producción.
2. **Volver a M4 con más regularización / un espacio de búsqueda de Optuna más
   conservador** (menos `num_leaves`, más `reg_alpha`/`reg_lambda`), apostando a
   que el modelo actual está sobreajustado al CV-pool.
3. **Revisar si hay margen para más pozos de entrenamiento** — con solo 4 pozos en
   CV-pool, es estructuralmente difícil que cualquier modelo flexible generalice
   bien; esto no se resuelve con más tuning si el límite real es la cantidad de
   pozos.

No se decide acá cuál — es una decisión de diseño para el usuario, no un detalle de
implementación.

## Diagnóstico de la brecha CV-test

Generado por [`ml/training/diagnose_cv_gap.py`](../ml/training/diagnose_cv_gap.py)
(`python -m ml.training.diagnose_cv_gap`), guardado en
[`docs/m4_diagnostics.json`](m4_diagnostics.json). **No reentrena ni reemplaza los
modelos de M4** — el diagnóstico 1 usa los hiperparámetros ya tuneados, leídos
directamente de la corrida logueada en MLflow (no retuneados, no copiados a mano); el
diagnóstico 2 sí corre una tuneada nueva, pero de un modelo *distinto* (sin las 4
features de ventana), no una nueva versión del modelo ya reportado arriba.

### 1. MAE de LightGBM por fold LOWO-CV — no solo el promedio de 64%

| Pozo held-out | Filas de validación | MAE LightGBM | MAE dummy (mismo fold) | Mejora vs. dummy | `best_iteration` |
|---|---|---|---|---|---|
| 1 | 6,389 | **1.65** | 33.37 | **95.1%** | 1000 (tope, sin early stopping) |
| 2 | 47,645 | 10.01 | 11.31 | 11.5% | 97 |
| 4 | 51,708 | 8.01 | 13.85 | 42.2% | 173 |
| 6 | 7,851 | 11.48 | 27.56 | 58.3% | 65 |

(`per_fold_cv_mae_mean_check` = 7.78778198504763, idéntico al `cv_mae_lowo` ya
logueado — confirma que el desglose reproduce exactamente el promedio original, no
un número distinto.)

**Sí había una señal visible antes del test final, tal como se sospechaba.** El
promedio de 64% de mejora está dominado por un solo fold extremo: con el pozo 1
como validación, LightGBM logra una mejora del 95.1% (MAE=1.65) y además agota las
1000 iteraciones permitidas sin que el early stopping lo frene — un comportamiento
atípico frente a los otros tres folds (65-173 iteraciones), compatible con un fold
inusualmente fácil de predecir a partir de los otros 3 pozos, no necesariamente
representativo del resto. Sacando ese fold, la mejora real en los otros tres
(11.5%, 42.2%, 58.3%) es bastante más modesta — y **el fold del pozo 2 (11.5%) es el
más parecido, en magnitud, a lo que terminó pasando en el test final** (donde
LightGBM prácticamente empata o pierde contra el dummy). El promedio agregado de CV
escondía esta señal detrás de un solo fold favorable.

### 2. Ablation: LightGBM sin las 4 features de ventana

| Métrica | Completo (16 features) | Sin ventana (12 features) | Diferencia |
|---|---|---|---|
| CV MAE (LOWO) | 7.79 | 7.93 | +1.8% peor en CV (marginal) |
| Test MAE pooled | 11.59 | **11.35** | 2.1% mejor sin ventana |
| Test MAE — dominante | 10.59 | **10.46** | 1.2% mejor sin ventana |
| Test MAE — atípico (pozo 0) | 16.83 | **15.98** | 5.1% mejor sin ventana |
| Test MAE — pozo 3 | **9.29** | 9.31 | 0.2% peor sin ventana (ruido, no señal) |
| Test MAE — pozo 5 | 14.31 | **13.76** | 3.8% mejor sin ventana |

**Evidencia parcial a favor de la hipótesis, pero no la explica todo.** Sacar las 4
features de ventana casi no cuesta nada en CV (+1.8%, dentro de ruido esperable) y
mejora el test en 4 de 5 desgloses (pooled, ambos régimen, 2 de 3 pozos) — consistente
con que esas features estén capturando algo de patrón específico del CV-pool que no
transfiere, tal como se sospechaba. Pero **el modelo sin features de ventana
tampoco le gana al dummy** (11.35 vs. 10.77 pooled, todavía 5.4% peor) — así que las
features de ventana son parte del problema, no la causa completa. El grueso de la
brecha CV-test persiste incluso con el feature set más simple.

### Conclusión de este diagnóstico

Ambos resultados apuntan en la misma dirección: **el problema no es un exceso de
features, es la heterogeneidad entre pozos combinada con muy pocos pozos de
entrenamiento (4).** El fold "fácil" (pozo 1) infló el promedio de CV de forma que
no era visible sin desagregar por fold — y aun quitando la fuente más obvia de
sobreajuste (las features de ventana), persiste una brecha de generalización de
~5% contra el dummy. Esto pesa a favor de la opción 3 del reporte original (revisar
si hay margen para más pozos) por sobre la opción 2 (más regularización) como única
respuesta — la regularización podría ayudar con la fracción de la brecha explicada
por las features de ventana, pero no con la fracción explicada por tener solo 4
pozos de referencia. Sigue siendo una decisión del usuario, no resuelta acá.

## Hipótesis 3: sobreajuste al protocolo de CV

Generado por `ml.training.diagnose_cv_gap.main_hypothesis_3()`
(`python -m ml.training.diagnose_cv_gap`), guardado en
[`docs/m4_diagnostics.json`](m4_diagnostics.json). **No reentrena ni reemplaza
ningún modelo de M4** — 3a entrena un modelo adicional puramente diagnóstico con
hiperparámetros fijos (no Optuna); 3b reconstruye la corrida de Optuna de M4 con la
misma semilla/búsqueda ya usada (determinística, no una búsqueda nueva) solo para
extraer el detalle por-fold que no se había guardado la primera vez.

### 3a. LightGBM con hiperparámetros fijos conservadores (sin Optuna)

Hiperparámetros usados (mismas 16 features que el modelo completo de M4):
`max_depth=4`, `num_leaves=15`, `min_child_samples=1000`, `colsample_bytree=0.75`,
`subsample=0.75`, `subsample_freq=1`. `learning_rate=0.05` no estaba en la
especificación del experimento — se completó con un valor moderado para no
confundir la comparación.

| Modelo | CV MAE (LOWO) | Test pooled | Test dominante | Test atípico (pozo 0) |
|---|---|---|---|---|
| Dummy | 21.52 | **10.77** | **9.64** | 16.64 |
| **Conservador fijo (sin Optuna)** | 9.90 | 11.24 | 10.06 | 17.37 |
| Ablation sin ventana (Optuna) | 7.93 | 11.35 | 10.46 | 15.98 |
| Completo tuneado (Optuna) | **7.79** | 11.59 | 10.59 | 16.83 |
| B&Y reducido | 14.07 | 11.85 | 12.14 | **10.32** |

**Patrón claro y monótono: a menos flexibilidad, más cerca del dummy en test —
pero ninguna versión de LightGBM lo supera en pooled.** Ordenando por MAE de test
pooled: dummy (10.77) < conservador (11.24) < ablation (11.35) < tuneado (11.59) <
B&Y reducido (11.85). El modelo conservador es 4.4% peor que el dummy en pooled
(y en cada régimen, casi la misma proporción: -4.4% dominante, -4.4% atípico) —
mejor que el tuneado (+3.1% con respecto a él) pero no suficiente para cruzar la
línea. Excepción notable: en el pozo 5 específicamente, el conservador sí le gana
al dummy (+8.1%) — no es una mejora uniforme, es más fuerte en algunos pozos que
en otros.

**Detalle que refuerza la hipótesis del pozo 1 como anómalo:** incluso con
hiperparámetros deliberadamente conservadores, el fold del pozo 1 volvió a agotar
casi todo el presupuesto de iteraciones (998 de 1000, contra 47-446 en los otros
tres folds) — el comportamiento atípico de ese fold no depende de qué tan agresivos
sean los hiperparámetros, algo en ese fold específico hace que el modelo "siga
mejorando" mucho más tiempo que en los demás.

### 3b. ¿Dominó el pozo 1 la selección de hiperparámetros de Optuna?

Reconstruido el detalle por-fold de los 15 trials de la ronda de tuning de M4
(misma semilla, mismos resultados — `winning_trial_value=7.78778...`, idéntico al
ya logueado):

| Pozo held-out | MAE medio entre los 15 trials | Desvío estándar | Rango (min-max) | Correlación con el ranking general del trial |
|---|---|---|---|---|
| **1** | 3.13 | **1.73** (55% del medio) | **1.07 – 7.59** (rango 6.51) | **0.80** |
| 2 | 9.97 | 0.57 (6% del medio) | 8.52 – 10.86 (rango 2.34) | -0.31 |
| 4 | 8.09 | 0.47 (6% del medio) | 7.18 – 9.22 (rango 2.05) | 0.50 |
| 6 | 12.81 | 0.58 (5% del medio) | 11.48 – 13.66 (rango 2.18) | 0.46 |

**Sí, claramente.** El fold del pozo 1 tiene una variabilidad relativa (55% de su
propia media) diez veces mayor que los otros tres folds (5-6%), y es el único con
una correlación fuerte con el ranking general que usa Optuna para elegir el
"ganador" (0.80, contra 0.46-0.50 de los otros dos positivos y -0.31 del pozo 2).
En la práctica: qué tan bien le va a un trial en el pozo 1 explica, por sí solo,
gran parte de qué trial termina eligiendo Optuna — no es un promedio balanceado
entre 4 folds igual de informativos. La correlación negativa del pozo 2 (-0.31) es
otra pieza de evidencia: sugiere que las configuraciones que mejor le achican el
error al pozo 1 tienden a costar un poco de performance en el pozo 2 — un
trade-off real entre folds, no ruido.

### Conclusión de Hipótesis 3

Las dos piezas de evidencia son consistentes entre sí y con el diagnóstico anterior:
el proceso de selección de hiperparámetros de M4 **sí estaba sesgado hacia el fold
del pozo 1**, y ese sesgo explica buena parte de por qué el modelo tuneado lucía
mucho mejor en CV de lo que resultó ser en test. Pero el experimento 3a muestra que
esto **no es solo un problema de regularización**: incluso el modelo más
conservador posible dentro de lo pedido sigue sin superar al dummy en pooled. La
regularización ayuda (mueve el resultado en la dirección correcta, de forma
monótona y medible) pero no alcanza para cerrar la brecha — consistente con que la
causa de fondo sea la cantidad de pozos disponibles para entrenar (4), no la
complejidad del modelo por sí sola. Ninguna de las 3 opciones del reporte original
se implementa acá — sigue siendo una decisión del usuario, ahora con más evidencia
para tomarla.

## Experimento de régimen (heterogeneidad de régimen en folds chicos, no solo "pocos pozos")

Generado por [`ml/training/diagnose_regime_experiment.py`](../ml/training/diagnose_regime_experiment.py)
(`python -m ml.training.diagnose_regime_experiment`), guardado en
[`docs/m4_regime_experiment.json`](m4_regime_experiment.json). No reentrena ni
reemplaza ningún modelo de M4.

### 1a. LightGBM + feature de régimen (`regime_score`)

`regime_score` = P(régimen dominante | MD, HD), de una regresión logística chica
fiteada **solo con datos del CV-pool** (nunca test), usando únicamente `MD` y `HD`
— ambas disponibles en tiempo real durante la perforación (siempre se sabe la
profundidad actual y el diámetro de broca/hoyo en uso), a diferencia de `well_id`,
que no tiene significado para un pozo genuinamente nuevo.

**El clasificador separa el CV-pool con 100% de exactitud** — hay un salto limpio de
MD entre los pozos atípicos del CV-pool (1, 6: máximo 634 m) y los dominantes (2, 4:
mínimo 988 m), sin superposición. Esto ya es una pista de lo que sigue.

Con esta feature agregada (17 features, mismo protocolo Optuna + LOWO-CV de 4 folds):
CV MAE=8.30, test pooled=**11.29** — mejor que el modelo completo tuneado (+2.6%)
pero todavía peor que el dummy (-4.9%). Mejora más en el régimen atípico
(pozo 0: 16.19, +2.7% vs. dummy, el mejor resultado de un modelo *único* — no
especializado — en ese pozo) que en el dominante (10.35, sigue detrás del dummy).

### 1b. Dos modelos separados por régimen — el resultado más fuerte de todos los diagnósticos, con una advertencia crítica

Un modelo entrenado **solo** con los pozos dominantes del CV-pool (2, 4; 99,353
filas; Optuna + LOWO-CV de 2 folds) evaluado solo contra los pozos dominantes de
test (3, 5). Otro modelo entrenado **solo** con los pozos atípicos del CV-pool
(1, 6; 14,240 filas; 2 folds) evaluado solo contra el pozo atípico de test (0).
Esto asume que ya se sabe a qué régimen pertenece cada fila de test — un supuesto
"oráculo" que se pone a prueba después.

| | CV MAE (2 folds) | Test (régimen propio) | vs. dummy (mismo régimen) | vs. mejor resultado previo en ese régimen |
|---|---|---|---|---|
| Modelo dominante (pozos 2,4 → test 3,5) | 9.44 | **9.91** | -2.8% (todavía peor, pero el más cerca visto) | mejor que todas las variantes previas de LightGBM (conservador 10.06, regime_score 10.35, ablation 10.46, tuneado 10.59) |
| Modelo atípico (pozos 1,6 → test 0) | 5.19 | **10.03** | **+39.7%** (le gana al dummy por mucho) | **+2.9% mejor que B&Y reducido** (10.32, el mejor hasta ahora en ese pozo) |

Por pozo individual, el modelo dominante le gana al dummy en el pozo 5 (13.51 vs.
13.82, +2.3%) aunque sigue perdiendo en el pozo 3 (8.65 vs. 8.18, -5.7%). El modelo
atípico es, sin excepción, el mejor resultado de todo M4 en el pozo 0 — mejor que
B&Y reducido, que hasta ahora era el único modelo que le ganaba al dummy ahí.

**Combinando ambos con el régimen real (conocido, no predicho) de cada pozo de
test**, el sistema completo da un MAE pooled de **9.93 — 7.8% mejor que el dummy**.
Es el único enfoque, de todos los diagnósticos corridos hasta ahora, que le gana al
dummy en pooled.

**La advertencia crítica: ese 7.8% de mejora asume un router perfecto, y el router
real (el clasificador de 1a) no lo es.** Aplicando el clasificador de `regime_score`
para decidir, fila por fila, a qué modelo especializado mandar cada fila de test
(en vez de usar la tabla de régimen conocida de ADR-003):

| Pozo | Exactitud del clasificador |
|---|---|
| 3 (dominante) | 100% |
| 5 (dominante) | 100% |
| **0 (atípico)** | **16.2%** |

El clasificador manda el 83.8% de las filas del pozo 0 al modelo *dominante* — el
que nunca vio nada parecido a un pozo atípico. La razón exacta ya estaba anticipada
en el análisis de ADR-003 y no se tuvo en cuenta al diseñar el experimento: el pozo
0 llega hasta MD=1206m, que **se superpone** con el rango de MD de los pozos
dominantes del CV-pool (desde 988m) — la separación "limpia" de 100% en el CV-pool
fue un artefacto de que los únicos 2 pozos atípicos del CV-pool (1, 6) resultan ser
particularmente someros (máximo 634m), no una propiedad robusta del régimen
atípico en general.

**Con el routing real del clasificador, el sistema combinado da MAE pooled = 12.01
— 11.5% PEOR que el dummy**, y catastrófico específicamente en el pozo 0 (22.96,
el peor resultado de cualquier modelo en cualquier pozo de todo M4) por el
enrutamiento mayoritariamente equivocado.

### Conclusión del experimento de régimen

Sí hay evidencia de que separar por régimen ayuda **al modelo en sí** — los
resultados oráculo son, con diferencia, los mejores vistos en todo M4, especialmente
en el pozo atípico. Pero el problema no es entrenar el modelo especializado, es
**enrutar correctamente un pozo nuevo a su régimen sin conocer su identidad** — y
el intento más simple de resolver eso (MD + HD con regresión logística) falla
específicamente en el caso que más importa (el pozo genuinamente atípico), porque
el CV-pool solo tenía 2 pozos atípicos y ambos eran someros por coincidencia, no
por definición del régimen. Esto no cierra la puerta a la idea de régimen — sugiere
que haría falta (a) más pozos atípicos en el CV-pool para que el router aprenda el
rango real de MD del régimen, no el de una muestra de 2, o (b) un router basado en
otras señales además de MD/HD. Ninguna alternativa se implementa acá.

### 2. ¿Vale la pena priorizar el enfoque de régimen sobre buscar más pozos?

Con la evidencia de arriba: **no de forma aislada.** El enfoque de régimen mejora
claramente el techo alcanzable (9.93 pooled con oráculo, mejor que cualquier cosa
vista), pero el cuello de botella pasó a ser el router, y el router falla
exactamente por el mismo motivo estructural que originó todo este diagnóstico:
muy pocos pozos por régimen en el CV-pool (2 dominantes, 2 atípicos) para
caracterizar la variabilidad real de cada régimen. Más pozos (opción 3 del reporte
original) ayudaría en ambos frentes a la vez — más pozos para entrenar el modelo
de cada régimen *y* más pozos para que el router generalice — mientras que quedarse
solo con el enfoque de régimen actual, sin más datos, no parece alcanzar para
producción.

## Reconocimiento del dataset completo de Volve (Tunkiel) — solo investigación, sin descargar

**No se pudo completar.** El servidor `ux.uis.no` devuelve `403 Forbidden` para
cualquier solicitud automatizada — confirmado con dos métodos independientes
(`WebFetch` y `curl` con user-agent de navegador), así que no es un bloqueo
específico de una herramienta, es el servidor rechazando el acceso no
interactivo/no institucional. No se descargó nada del dataset de 2.7GB, tal como
se pidió.

Lo que se pudo establecer por otras fuentes (con menor certeza, sin verificación
directa contra el archivo real):

- El dataset completo (Volve, parseado de WITSML a CSV por Tunkiel) cubre más
  pozos que los 7 de USROP — la existencia de una curación explícita en el paper
  de USROP (Tunkiel et al. 2021) implica que hubo pozos descartados por
  completitud/calidad, no que solo existan 7 pozos con datos de perforación en
  todo el campo Volve.
- Una fuente (paper "TADI", arXiv 2605.00060) menciona "sparser reporting" en
  pozos de la era de exploración del campo Volve frente a los de desarrollo
  (2007-2016) — consistente con la hipótesis de que hay pozos adicionales, pero
  con completitud desigual, no directamente comparables a los 7 ya curados.
- **No se pudo determinar un número concreto** de pozos adicionales con las 12
  columnas equivalentes y completitud similar — sería necesario browsear
  `file_list.html` desde un navegador real (posiblemente bloqueado por IP/región,
  no por ser un bot) o hacer una descarga liviana de solo metadata/listado de
  archivos (no el dataset completo) como un paso de reconocimiento separado y
  todavía barato, antes de comprometerse a procesar los 2.7GB completos.

No se puede usar esto para decidir la opción 3 del reporte original todavía — el
reconocimiento quedó incompleto, no negativo.

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

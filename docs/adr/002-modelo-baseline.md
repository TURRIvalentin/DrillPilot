# ADR-002: Modelo único (LightGBM) y baseline físico obligatorio para el MVP

## Estado
Aceptado

## Contexto
El diseño inicial de DrillPilot contemplaba comparar tres frameworks de
gradient boosting (LightGBM, XGBoost, CatBoost) como parte del MVP, más un
baseline "dummy" (media / regresión lineal) como piso de referencia.

Con el dataset ya fijado en ADR-001 (USROP, 7 pozos, ~200k muestras, 12
atributos), el cuello de botella del proyecto no es encontrar el mejor
framework de boosting — en datasets de este tamaño y con features
tabulares homogéneas, LightGBM, XGBoost y CatBoost suelen converger a
performance muy similar tras tuning, y la literatura de USROP ya usa
gradient boosting como referencia estándar. Correr y tunear tres
frameworks completos en el MVP consume presupuesto de tiempo del
proyecto sin mover significativamente la métrica de negocio, y contradice
la regla de "no escribir código innecesario" fijada en las reglas del
proyecto.

Por otro lado, el dominio de perforación tiene un modelo físico estándar
y ampliamente citado — Bourgoyne & Young (1974) — que predice ROP a
partir de WOB, RPM, profundidad y otras variables mediante una ecuación
de regresión multiplicativa con 8 coeficientes. Comparar el modelo de ML
únicamente contra un baseline dummy no responde la pregunta que le
importa a un ingeniero de perforación: "¿el ML mejora sobre lo que ya usa
la industria?". Un baseline dummy no responde eso; un baseline físico sí.

## Decisión
1. El MVP entrena y tunea **un único algoritmo: LightGBM**. XGBoost y
   CatBoost quedan explícitamente fuera de scope del MVP y se listan en
   el roadmap como "comparación opcional, post-MVP".
2. El mismo milestone que implementa el baseline dummy implementa
   también la **ecuación de Bourgoyne & Young** como segundo baseline,
   no-ML. El modelo final reporta su mejora relativa (MAE) contra ambos
   baselines, no solo contra el dummy.

## Alternativas consideradas
- **Comparar LightGBM, XGBoost y CatBoost en el MVP** (diseño original):
  descartada por costo de tiempo/mantenimiento (3 pipelines de tuning en
  vez de 1) sin evidencia de que la mejora de performance justifique ese
  costo en esta etapa del proyecto.
- **Solo baseline dummy, sin Bourgoyne & Young**: descartada porque no
  permite argumentar el valor del proyecto frente a un lector con
  background de perforación — la pregunta relevante es "vs. la industria",
  no "vs. la media".
- **Ensamble LightGBM+XGBoost+CatBoost directamente**: descartada por
  agregar complejidad de despliegue (tres modelos a servir/versionar en
  MLflow) sin que el problema lo requiera en esta fase.

## Consecuencias
- Positivas: pipeline de entrenamiento más simple y rápido de iterar;
  narrativa del proyecto más fuerte al comparar contra un estándar de
  industria real, no solo contra un piso estadístico; menos superficie
  de mantenimiento en el MVP.
- Negativas: no hay evidencia empírica propia, dentro del MVP, de que
  LightGBM sea superior a XGBoost/CatBoost en este dataset específico —
  esa validación queda diferida a la fase post-MVP listada en el roadmap.
- Bourgoyne & Young requiere fitear sus coeficientes por regresión sobre
  el mismo train set que el modelo de ML, respetando el split de
  [ADR-003](003-split-strategy.md) (CV-pool: pozos 1/2/4/6, leave-one-well-out)
  para que la comparación sea justa.
- **Resuelto en M4 (ver sección de mapeo más abajo)**: de los 8 términos de
  Bourgoyne & Young, 4 deben omitirse por completo (sin dato disponible en
  USROP ni proxy razonable) y 1 más requiere una constante no estandarizada
  en la literatura. Esto debilita materialmente la comparación "ML vs.
  estándar de industria" que motivó este ADR — ver el detalle y sus
  implicancias abajo, no se minimiza acá.

## Mapeo Bourgoyne & Young vs. USROP (M4)

Antes de implementar el baseline físico, se mapearon los 8 términos de la
ecuación de Bourgoyne & Young contra los 12 atributos de USROP y las 4
features derivadas del pipeline de M3.

**Nota sobre verificación de fuentes:** la estructura general del modelo
(8 subfunciones f1-f8, y la lista de variables de entrada que requiere:
profundidad D, WOB, RPM, gradiente de presión de poro gp, densidad
equivalente de lodo ρc, desgaste fraccional de diente h, fuerza de impacto
hidráulico Fj) se confirmó en esta sesión contra múltiples fuentes
secundarias independientes (no solo memoria de entrenamiento) — ver
Referencias. Las **constantes numéricas exactas** de cada término (valor de
referencia de profundidad, valor de referencia de gradiente de presión,
exponentes) están citadas de acuerdo a lo que aparece consistentemente en
la literatura de ingeniería de perforación (Bourgoyne, Millheim, Chenevert
& Young, 1986), pero no se pudieron re-verificar línea por línea contra el
texto original en esta sesión (las fuentes accesibles públicamente muestran
las ecuaciones como imágenes no extraíbles o están detrás de paywall). Se
marcan explícitamente abajo como "a verificar contra la fuente primaria
antes de hardcodear en el código" donde aplica — no se afirma una precisión
que no se pudo confirmar.

**Ecuación general:** `ROP = exp(a1 + Σ(i=2..8) ai·xi)`. `a1` es la
constante de resistencia de formación (intercept); `a2..a8` son los
coeficientes que se van a fitear por regresión sobre el mismo train set
que LightGBM (CV-pool, pozos 1/2/4/6), no valores de literatura.

| Término | Efecto que modela | Fórmula | Categoría | Detalle |
|---|---|---|---|---|
| `a1` | Resistencia/drillability de formación (constante) | Intercept de la regresión | **1 — estimable** (como coeficiente, no como columna) | Se fitea junto con `a2..a8` sobre el train set; no requiere una columna de USROP ni una constante de literatura. |
| `x2` | Profundidad / tendencia de compactación normal | `x2 = 10,000 − D` (D en ft) | **1 — estimable directamente** | `D` = `MD` (o `VD`) de USROP. Requiere conversión de unidad (USROP está en metros, la fórmula original usa ft) — detalle de implementación, no bloqueante. |
| `x3` | Presión de poro / sub-compactación (presión anormal) | `x3 = D^0.69 · (gp − 9.0)` (gp en ppg) | **3 — omitir** | USROP no tiene gradiente de presión de poro (`gp`) ni proxy (sin resistividad, sin sónico, sin presión medida). Fijar `gp = 9.0` (gradiente normal) haría `x3 = 0` siempre — matemáticamente equivalente a omitir el término para cualquier `a3`, así que se documenta como omisión, no como "constante fijada". |
| `x4` | Presión diferencial / chip hold-down | `x4 = D · (gp − ρc)` | **3 — omitir** | Mismo problema que `x3` (`gp` no disponible). `ρc` (densidad de lodo) sí está disponible (`DS`, con conversión g/cm³→ppg), pero no alcanza para calcular el término sin `gp`. |
| `x5` | Peso sobre broca por pulgada de diámetro (ajustado por umbral) | `x5 = ln[(W/db − (W/db)ₜ) / (4 − (W/db)ₜ)]` | **2 — mixta: directa + constante** | `W` = `WOB`, `db` ≈ `HD` (diámetro de hoyo como proxy de diámetro de broca), ambos disponibles con conversión de unidad. `(W/db)ₜ` (peso umbral) es específico de tipo de broca/formación — la propia literatura no da un valor único universal. Fuente típica: Bourgoyne, Millheim, Chenevert & Young (1986), cap. 6, ejemplos resueltos — **a verificar contra la fuente antes de hardcodear**. Alternativa a evaluar en M4: tratar `(W/db)ₜ` como hiperparámetro a ajustar por CV en vez de una constante fija de literatura, ya que no hay un "valor típico" defendible sin calibración específica. |
| `x6` | Velocidad rotatoria | `x6 = ln(N/60)` (N en rpm) | **1 — estimable directamente** | `N` = `RPM` de USROP, sin conversión adicional más allá de la que ya trae la fórmula. |
| `x7` | Desgaste de diente de broca | Función de `h` (desgaste fraccional 0–1), fórmula depende del tipo de broca | **3 — omitir** | USROP no tiene ningún registro de broca (sin ID de corrida, sin dull grade, sin historial de desgaste). No hay proxy razonable dentro del dataset. |
| `x8` | Hidráulica de broca / fuerza de impacto del chorro | `x8 = ln(Fj / 1000)`, `Fj` = f(caudal, densidad de lodo, área de toberas) | **3 — omitir** | `FR` (caudal) y `DS` (densidad) están disponibles, pero falta el área total de toberas (TFA) de la broca, que no está en USROP y — a diferencia de `x5` — tampoco tiene un "valor típico" razonable en literatura (varía por selección de tobera en cada corrida específica). |

**Features derivadas de M3 (`WOB_rolling_mean`, `RPM_rolling_mean`,
`T_rolling_std`, `WOB_diff_1`): ninguna mapea a un término de Bourgoyne &
Young.** El modelo es instantáneo/multiplicativo por diseño (usa el valor
puntual de cada variable en el momento, no una tendencia de ventana) — no
es un error de mapeo, es una diferencia estructural entre ambos enfoques
que vale la pena tener presente al comparar sus resultados.

### Impacto en la validez de la comparación (sin minimizar)

**4 de los 8 términos (`x3`, `x4`, `x7`, `x8`) se omiten por completo**, y
un quinto (`x5`) requiere una constante sin valor único defendible en la
literatura. Lo que se va a implementar en M4 como "baseline Bourgoyne &
Young" es, en los hechos, **un modelo reducido de 3-4 términos utilizables
(`a1`, profundidad, RPM, y WOB con una constante incierta)**, no la
ecuación completa de 8 términos que se usa como referencia en la
industria.

Esto debilita específicamente la narrativa "¿el ML mejora sobre lo que ya
usa la industria?" que motivó este ADR: la omisión más costosa es `x7`
(desgaste de broca), un driver reconocido de la caída de ROP dentro de una
misma corrida — al no estar disponible, cualquier señal de desgaste que
exista en los datos va a terminar absorbida (mal atribuida) por otras
variables correlacionadas (profundidad, WOB, tiempo) tanto en la regresión
de Bourgoyne & Young como, de forma más flexible, en LightGBM. La ausencia
de presión de poro (`x3`, `x4`) y de hidráulica (`x8`) también deja fuera
mecanismos físicos reales que en pozos con presión anormal o hidráulica
pobre explican una fracción no trivial del comportamiento de ROP.

**Conclusión honesta:** el baseline que se va a implementar no es "el
estándar de la industria" sin salvedades — es la mejor aproximación
posible al estándar de la industria **dado lo que USROP permite medir**.
Se recomienda que cualquier reporte final (README, evaluación de M8) llame
a este baseline explícitamente "Bourgoyne & Young reducido (4/8 términos,
limitado por las variables disponibles en USROP)", no "Bourgoyne & Young"
a secas, para no sobrevender el rigor de la comparación.

## Referencias
- Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting
  Decision Tree." NeurIPS.
- Bourgoyne, A.T. Jr. & Young, F.S. Jr. (1974). "A Multiple Regression
  Approach to Optimal Drilling and Abnormal Pressure Detection."
  SPE Journal.
- Bourgoyne, A.T. Jr., Millheim, K.K., Chenevert, M.E. & Young, F.S. Jr.
  (1986). *Applied Drilling Engineering*. SPE Textbook Series, Vol. 2,
  cap. 6 — fuente típica de las constantes numéricas y ejemplos resueltos
  del modelo; **constantes exactas a re-verificar contra esta fuente antes
  de hardcodear en el código** (ver nota de verificación arriba).
- Estructura general del modelo (8 subfunciones, variables de entrada)
  confirmada en esta sesión contra: Hindawi/Wiley, *Mathematical Modeling
  Applied to Drilling Engineering* (Nascimento et al., 2015); Scialert,
  *Determining Bourgoyne and Young Model Coefficients Using Genetic
  Algorithm* (2008); búsqueda web general — ninguna con acceso al texto
  completo de las ecuaciones (imágenes no extraíbles o paywall).
- ADR-001: Dataset selection (USROP).
- [docs/adr/003-split-strategy.md](003-split-strategy.md) — split por pozo
  que también aplica al fiteo de `a1..a8`.

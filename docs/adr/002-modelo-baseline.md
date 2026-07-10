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
- Bourgoyne & Young requiere fitear sus 8 coeficientes por regresión
  sobre el mismo train set que el modelo de ML, manteniendo el mismo
  split por pozo (GroupKFold) para que la comparación sea justa.
- **Pendiente antes de M4**: mapear los 8 términos de la ecuación de
  Bourgoyne & Young contra los 12 atributos de USROP, indicando cuáles
  son estimables directamente, cuáles requieren fijar un valor constante
  de literatura (citado), y cuáles hay que omitir.

## Referencias
- Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting
  Decision Tree." NeurIPS.
- Bourgoyne, A.T. Jr. & Young, F.S. Jr. (1974). "A Multiple Regression
  Approach to Optimal Drilling and Abnormal Pressure Detection."
  SPE Journal.
- ADR-001: Dataset selection (USROP).

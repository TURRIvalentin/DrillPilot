# ADR-003: Estrategia de split (test final + CV) para USROP

## Estado
Propuesto

## Contexto

USROP tiene solo 7 pozos (198,928 filas). Un `GroupKFold` genérico (ej. 5
folds arbitrarios sobre 7 grupos) da particiones desbalanceadas en tamaño y
en composición, y el resultado queda dominado por qué pozo específico cae
en qué fold, no por la varianza real del modelo. Con tan pocos grupos, la
elección de qué pozo(s) quedan como test final no puede ser arbitraria ni
aleatoria: si el pozo de test resulta ser atípico respecto al resto, la
métrica final queda distorsionada y no dice nada confiable sobre
generalización.

Se perfilaron los 7 pozos con datos reales (no supuestos) para fundamentar
la decisión:

| Pozo | Filas | % del total | MD (m) | Formación | ROP media | ROP std | RPM std |
|---|---|---|---|---|---|---|---|
| 0 | 13,746 | 6.91% | 491–1,206 (span 715) | N-NA | 39.10 | 11.97 | 41.56 |
| 1 | 6,389 | 3.21% | 301–634 (span 332) | N-S | 55.27 | 16.03 | 39.59 |
| 2 | 47,645 | 23.95% | 988–3,466 (span 2,478) | N-SH | 24.57 | 13.29 | 35.34 |
| 3 | 53,041 | 26.66% | 1,307–4,065 (span 2,759) | N-SH | 21.58 | 9.63 | 19.35 |
| 4 | 51,708 | 25.99% | 1,401–4,090 (span 2,690) | N-SH | 17.33 | 8.87 | 59.16 |
| 5 | 18,548 | 9.32% | 2,828–3,792 (span 964) | N-SH | 26.48 | 15.92 | 75.93 |
| 6 | 7,851 | 3.95% | 225–634 (span 408) | N-SH | 47.90 | 20.03 | 37.08 |

**Hallazgo de heterogeneidad (no es solo el código de formación):** agrupar
por el prefijo de formación (`N-SH` en 5 de 7 pozos) es engañoso si se usa
solo. Los pozos se separan más claramente en dos regímenes reales:

- **Pozos 0, 1 y 6** — MD corto y somero (225–1,206 m, span 332–715 m),
  ROP media alta (39–55 m/h). El pozo 6 tiene código `N-SH` pero se
  comporta como los pozos 0 y 1 (`N-NA`, `N-S`) en profundidad y en ROP,
  no como los pozos 2/3/4/5. Agruparlo con estos últimos solo por el
  prefijo del nombre habría sido un error.
- **Pozos 2, 3, 4 y 5** — MD largo y profundo (988–4,090 m, span
  964–2,759 m), ROP media baja-a-media (17–26 m/h). Representan el 85.9%
  de las filas del dataset (170,942 de 198,928).

El pozo 5, dentro de este segundo grupo, solo cubre el tramo profundo
(2,828–3,792 m) — no tiene datos del tramo somero que sí tienen 2/3/4 — y
tiene la mayor variabilidad de RPM de los 7 pozos (std 75.93), así que
tampoco es un representante perfectamente "promedio" del grupo, aunque
comparte régimen de ROP y profundidad con 2/3/4.

**Riesgo concreto que esto confirma:** si el test final terminara siendo
únicamente el pozo 1 o el pozo 6 (los de ROP más alto, pozos más cortos),
el modelo se evaluaría contra un solo pozo que representa apenas 3.21% o
3.95% del dataset individualmente — y contra un régimen de perforación
(pozos 0+1+6 combinados, 14.1% del total) sistemáticamente distinto del
resto — exactamente el escenario que este ADR busca evitar.

## Decisión

**1. Test final: pozos 0, 3 y 5.** Reservados y no tocados hasta la
evaluación terminal (ni en tuning, ni en ninguna corrida de CV). Los pozos
3 y 5 pertenecen al régimen dominante ("profundo, ROP bajo-medio", 85.9%
de las filas). El pozo 0 se agrega deliberadamente para representar el
régimen atípico ("somero, ROP alto", 14.1% de las filas) — un test final
que excluye por completo un régimen que existe en producción no responde
si el modelo generaliza a él, y esa es una pregunta que el proyecto tiene
que poder contestar, no evitar. Combinados suman 85,335 filas (42.9% del
total) — una fracción de test todavía más grande que la ya elevada de la
versión anterior de este ADR (36.0%), consecuencia directa de exigir
representación de ambos regímenes con solo 7 pozos disponibles.

### Por qué 3 y 5 específicamente dentro del régimen dominante

Criterio explícito y reproducible: dentro del régimen dominante (pozos 2,
3, 4, 5), se ordenan por tamaño y se toman los dos extremos — el más
grande y el más chico.

- **Pozo 3 (53,041 filas, 26.66%)** — el más grande, maximiza el poder
  estadístico del test. **Honestidad sobre este criterio:** la diferencia
  con el pozo 2 (47,645, 23.95%) y el pozo 4 (51,708, 25.99%) es chica —
  no hay una señal en los datos de este ADR que distinga cualitativamente
  al pozo 3 de 2 o 4 más allá de ser, por poco margen, el más grande.
  Elegir pozo 2 o pozo 4 en su lugar sería igual de defendible; no se
  afirma que 3 sea "mejor" que 2 o 4, solo que el criterio de tamaño lo
  selecciona.
- **Pozo 5 (18,548 filas, 9.32%)** — el más chico del régimen dominante,
  para minimizar cuánto volumen de entrenamiento se pierde al reservarlo:
  agregar un segundo pozo grande (ej. 2 o 4) al test, sumado al pozo 0,
  hubiera dejado un pool de CV demasiado chico para tuning confiable.
  Adicionalmente (observación posterior, no el criterio que motivó la
  elección) el pozo 5 cubre únicamente el tramo profundo (2,828–3,792 m),
  a diferencia de 2/3/4 que cubren desde ~1,000-1,400 m — da algo de
  diversidad de profundidad al test en vez de ser redundante con el pozo 3.

### Por qué el pozo 0 específicamente dentro del régimen atípico

Mismo criterio de tamaño aplicado al régimen atípico (pozos 0, 1, 6): se
toma el más grande, que es el pozo 0 (13,746 filas, 6.91%, contra 3.21%
del pozo 1 y 3.95% del pozo 6) — maximiza el poder estadístico de la
porción de test que mide el régimen minoritario, evitando que ese
desglose quede basado en una muestra todavía más chica de lo necesario.

**Trade-off que esto implica, explícito:** el pozo 0 es, de los tres
atípicos, el que menos se aleja del régimen dominante (ROP media 39.10,
contra 47.90 del pozo 6 y 55.27 del pozo 1 — el régimen dominante está en
17-26). Priorizar tamaño sobre "qué tan atípico es" hace que el desglose
por régimen atípico sea más confiable estadísticamente pero, en los
hechos, un poco menos exigente como prueba de generalización que si se
hubiera elegido el pozo 1 (el caso más extremo, pero con solo 3.21% de
los datos — una estimación mucho más ruidosa). Es una decisión consciente
de priorizar confiabilidad de la métrica sobre severidad de la prueba; se
documenta acá para que quede trazable, no implícita.

### Cómo se van a reportar las métricas finales

No un solo MAE pooled. La evaluación final (milestone de evaluación, no
este ADR) tiene que reportar, como mínimo:

1. **MAE pooled** sobre las 85,335 filas de test (los 3 pozos juntos) —
   número headline.
2. **MAE por pozo individual** (pozo 0, pozo 3, pozo 5) — 3 números,
   visibilidad total, sin promediar nada todavía.
3. **MAE por régimen**: dominante (pozos 3+5 combinados, 71,589 filas)
   vs. atípico (pozo 0, 13,746 filas) — el desglose que responde
   directamente si el modelo generaliza al régimen minoritario, que es la
   pregunta que un solo número pooled no puede contestar.

Los mismos tres niveles (pooled, por pozo, por régimen) se calculan
también para los baselines dummy y Bourgoyne & Young de ADR-002, para que
la comparación de mejora relativa del modelo final sea honesta por
régimen y no se esconda detrás de un promedio dominado por el régimen
mayoritario.

**2. CV para tuning: leave-one-well-out (LOWO) sobre los 4 pozos
restantes (1, 2, 4, 6).** Cada fold entrena con 3 pozos y valida contra
el pozo restante, rotando los 4. Esto sigue incluyendo pozos atípicos (1,
6) en el pool de CV — a diferencia del test final, acá no hay problema en
que un fold sea "raro": el score de CV se promedia entre los 4 folds, así
que la rareza de un pozo puntual se diluye en el promedio, no domina el
resultado como pasaría con un test set de un solo pozo. Esto también
expone al proceso de tuning a la heterogeneidad real entre regímenes de
perforación, en vez de ocultarla.

**Nota sobre el tamaño del pool de CV:** 4 pozos (antes eran 5) es un
mínimo ajustado para leave-one-well-out. Se consideró agregar un segundo
pozo atípico al test (ver Alternativas) para reforzar también la
robustez del desglose por régimen atípico, pero se descartó porque
hubiera dejado el pool de CV en solo 3 pozos — insuficiente para tuning
razonable. Es la razón concreta por la que el test final tiene un solo
pozo atípico y no dos.

**3. Ninguna ventana temporal (rolling, lag) puede cruzar el límite entre
pozos.** Cada fila del dataset debe llevar una columna `well_id` explícita
(derivada del nombre de archivo) desde la carga inicial. Cualquier
transformer de `ml/features/` que calcule rolling/lag debe:

- Recibir `well_id` como parámetro obligatorio, no opcional.
- Aplicar `groupby("well_id", sort=False)` **antes** de cualquier
  `.rolling(...)` o `.shift(...)` — nunca calcular la ventana sobre el
  DataFrame concatenado directamente.
- Venir acompañado de un test unitario que construya un DataFrame
  sintético con 2+ pozos donde cruzar el límite daría un resultado
  distinto (y detectablemente incorrecto) del que respeta el límite, y
  verifique que el transformer produce el resultado que respeta el
  límite. No es un detalle de implementación opcional: es un requisito de
  test por cada feature de ventana temporal que se agregue en el
  milestone siguiente.

## Alternativas consideradas

1. **`GroupKFold` genérico (ej. `sklearn.model_selection.GroupKFold(n_splits=5)`
   sobre los 7 pozos).** Descartada — es exactamente el problema que
   motiva este ADR: con 7 grupos de tamaño muy desigual (3.21% a 26.66%
   del dataset), los folds quedan desbalanceados y el resultado depende
   más de qué pozo cae en qué fold que de la varianza real del modelo.

2. **Test final = un único pozo (el pozo 5, el más chico del grupo
   "típico").** Reduce el costo en datos de entrenamiento (solo se
   perdería 9.32% del dataset en vez de 42.9%), pero dejaría el test final
   sujeto a la idiosincrasia de un solo pozo — el mismo problema de
   varianza alta que llevó a preferir varios pozos, solo que con un pozo
   ya identificado como no perfectamente representativo del grupo (única
   cobertura del tramo profundo, RPM más variable de los 7).

3. **Test final = solo pozos del régimen dominante (3 y 5), sin ningún
   pozo atípico — versión anterior de este ADR.** Descartada tras
   revisión: un test que excluye por completo el régimen atípico (0, 1,
   6) nunca puede responder si el modelo generaliza a él, que es
   exactamente la pregunta que separar por régimen busca contestar. Se
   reemplaza por la decisión actual (pozos 0, 3 y 5).

4. **Test final = pozos 3, 5 y dos pozos atípicos (ej. 0 y 6), en vez de
   uno solo.** Daría un desglose por régimen atípico más robusto
   (13,746+7,851 filas en vez de solo 13,746), pero dejaría el pool de CV
   en solo 3 pozos (1, 2, 4) — insuficiente para leave-one-well-out
   razonable. Descartada por ese motivo, no porque un segundo pozo
   atípico no aportara valor — es un trade-off explícito entre robustez
   del desglose de test y robustez del tuning, resuelto a favor del
   tuning porque sin un tuning razonable el número de test tampoco vale
   mucho.

5. **K-Fold aleatorio a nivel de fila (sin agrupar por pozo).**
   Descartada de plano — filas del mismo pozo en train y en
   validación/test simultáneamente es leakage directo (la razón original,
   ya establecida en el análisis de riesgos inicial del proyecto, de
   agrupar por pozo en primer lugar).

## Consecuencias

**Positivas**
- La decisión de qué pozos van a test queda trazable a datos reales
  (tabla de arriba), no a una asignación aleatoria de un `GroupKFold`
  genérico.
- El test final incluye ambos regímenes identificados (dominante y
  atípico) — el proyecto va a poder reportar generalización al régimen
  minoritario, no solo asumirla o evitarla.
- El desglose de métricas por régimen (pooled + por pozo + por régimen)
  queda especificado antes de implementar la evaluación, no como una
  ocurrencia tardía.
- La regla de no cruzar límites de pozo en ventanas temporales queda
  fijada *antes* de escribir el pipeline de features, con un mecanismo de
  enforcement concreto (test obligatorio por feature), no como una
  intención vaga.
- Los criterios de selección de pozos quedan documentados de forma
  explícita, incluyendo dónde el criterio es débil (pozo 3 vs. 2 vs. 4)
  en vez de implicar un análisis más fuerte del que realmente hubo.

**Negativas / riesgos**
- 42.9% de test final es una fracción muy grande — el pool de
  entrenamiento para tuning queda en solo 4 pozos (113,593 filas, 57.1%).
  Es consecuencia directa de exigir representación de ambos regímenes con
  solo 7 pozos disponibles, no de una mala decisión de split; se
  documenta como limitación estructural del dataset, no del proceso.
- El desglose por régimen atípico en el test final descansa en un solo
  pozo (pozo 0, 13,746 filas) — más robusto que dejarlo en 0 pozos (la
  versión anterior de este ADR), pero sigue siendo una muestra chica y
  además la menos extrema de los tres pozos atípicos disponibles (ver
  trade-off documentado arriba). El número de "MAE régimen atípico" hay
  que leerlo con esa salvedad, no como una medición robusta de todo el
  espacio de comportamientos atípicos.
- El pool de CV (1, 2, 4, 6) queda en 4 pozos — leave-one-well-out con 4
  folds es el mínimo razonable, no hay margen para perder ninguno más.
- Ningún pozo del test final (0, 3, 5) participa del LOWO-CV de tuning,
  así que las decisiones de hiperparámetros nunca ven esos tres pozos
  hasta la evaluación terminal — correcto por diseño, pero implica que el
  tuning está optimizando sobre un subconjunto de regímenes (1, 2, 4, 6)
  distinto del que finalmente se reporta.

## Referencias

- [docs/adr/001-dataset-selection.md](001-dataset-selection.md) — origen
  del riesgo de no-estacionariedad entre pozos que motiva este ADR.
- [docs/eda_findings.md](../eda_findings.md) — heterogeneidad entre pozos
  ya documentada en M2 (distribuciones de ROP/WOB/RPM/Torque/SPP por
  pozo, sin agregar).
- `docs/eda/zero_value_investigation.json` y el perfilado de este ADR —
  fuente de los números de la tabla de contexto.

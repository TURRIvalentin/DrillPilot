# ADR-004: Contrato de entrada de la API de inferencia

## Estado
Aceptado

## Contexto

`ml/inference/predict.py` (M6) expone `predict_rop(history, model)`, donde
`history` es un `DataFrame` con **varias filas ordenadas por MD** de un mismo
pozo/sesión, no una lectura aislada — no es una decisión arbitraria de la
implementación, es una consecuencia directa de cómo está construido el modelo: dos
de las 14 features (`WOB_rolling_mean_10`, `RPM_rolling_mean_10`, ver
`docs/feature_dictionary.md`) son medias móviles hacia atrás dentro del mismo pozo,
y no se pueden calcular correctamente a partir de una única lectura instantánea.

Antes de diseñar el endpoint de M7 (FastAPI) hay que decidir **quién es
responsable de mantener ese historial**: el cliente (que manda la ventana
completa en cada request) o el servicio (que la acumula internamente por
pozo/sesión). Esto no es un detalle de implementación — el diseño original del
proyecto fijó explícitamente que "el servicio es stateless" (ver el análisis de
arquitectura inicial); una de las dos opciones revierte esa decisión y necesita
justificación propia si se elige, no puede asumirse en silencio.

## Decisión

**El endpoint recibe la ventana de lecturas (últimas N, ordenadas por MD, de un
mismo pozo) en cada request. El cliente es responsable de mantener y enviar el
historial. El servicio permanece stateless**, consistente con el diseño original.

Contrato concreto:

- Mínimo aceptado: 1 fila (el pipeline no rechaza menos de `rolling_window`
  filas — usa ventana parcial, `min_periods=1`, ver `docs/feature_dictionary.md`).
- Recomendado: ≥10 filas (el `rolling_window` por defecto del transformer) para
  que las medias móviles no estén en su régimen degradado de "poco historial,
  poco suavizado" documentado en M3/M5.
- El endpoint devuelve una predicción por fila recibida (igual que
  `predict_rop` ya hace) — el cliente que solo quiere "el ROP actual" toma la
  última.
- `well_id`/identificador de sesión: cualquier valor consistente entre las
  filas de una misma request sirve, no necesita coincidir con ninguno de los 7
  pozos de entrenamiento (ya documentado en `ml/inference/predict.py`).

## Alternativas consideradas

### a) Cliente manda la ventana completa en cada request (elegida)

El endpoint es una función pura de su input — recibe N lecturas, devuelve N
predicciones, no guarda nada entre requests.

- **A favor:** consistente con el diseño stateless ya fijado, sin necesidad de
  justificar una reversión. Escala horizontalmente sin coordinación (cualquier
  instancia del servicio puede atender cualquier request, no hace falta
  "sticky sessions" ni un store de estado compartido). Sin riesgo de memoria
  creciendo con sesiones abandonadas. Más honesto sobre qué necesita el modelo
  — no esconde el requisito de historial detrás de una interfaz que sugiere
  "una lectura, una predicción" cuando eso no es lo que el modelo hace.
- **En contra:** el cliente tiene que mantener y mandar la ventana. En la
  práctica, el consumidor real de esta API (un sistema de adquisición de datos
  de perforación, WITSML o similar) ya tiene el stream continuo con historial
  — no es pedirle a un cliente "tonto" que haga algo ajeno a su dominio, es
  aprovechar que el historial ya existe del otro lado. El costo real es mayor
  para un cliente de prueba/demo simple (ej. un script que solo tiene "la
  lectura de ahora"), no para el caso de uso real del proyecto.

### b) Cliente manda una lectura, el servicio acumula estado por pozo/sesión

El endpoint recibe una fila, la agrega a un buffer interno identificado por
pozo/sesión, y predice usando ese buffer acumulado.

- **A favor:** interfaz más simple para un cliente que solo tiene la lectura
  actual.
- **En contra, y por qué se descarta:** revierte explícitamente la decisión de
  "servicio stateless" del diseño original, sin una razón de negocio que lo
  justifique en este proyecto (no hay un cliente real restringido a mandar
  solo una lectura por vez). Agrega problemas nuevos que no existen hoy:
  identificación de sesión/pozo persistente entre requests, política de
  expiración de buffers abandonados (¿cuándo se olvida un pozo que dejó de
  mandar datos?), pérdida de historial ante un restart del servicio
  (predicciones degradadas hasta rellenar el buffer de nuevo), y
  escalado horizontal que ya no es trivial (una request de "pozo X" tiene que
  llegar a la instancia que tiene el buffer de "pozo X", o todas las
  instancias tienen que compartir un store externo tipo Redis). Ninguno de
  estos problemas es imposible de resolver, pero son complejidad nueva para
  resolver un problema que el cliente real de este proyecto no tiene.

## Consecuencias

**Positivas**
- M7 (FastAPI) puede diseñarse stateless desde el arranque, sin deuda técnica
  de "agregar estado más adelante" ni la necesidad de un store externo.
- El contrato de la API es honesto sobre qué necesita el modelo — no hay
  sorpresas para quien la integre.
- Reutiliza tal cual el diseño ya validado de `ml/inference/predict.py`
  (M6) sin cambios.

**Negativas / riesgos**
- El payload de cada request es más pesado que "una lectura" (hasta N filas).
  Para el volumen de este proyecto (predicciones sobre series de perforación,
  no trading de alta frecuencia) no es un problema de performance real, pero
  queda anotado.
- Un cliente que integre esta API mal (mandando ventanas de 1 fila
  sistemáticamente) va a obtener predicciones degradadas silenciosamente —
  el pipeline no rechaza ventanas cortas, solo las procesa con menos
  suavizado. M7 debería documentar esto claramente en el endpoint (ej. en la
  respuesta o en el schema de OpenAPI), no asumir que quien integra leyó este
  ADR.

## Referencias

- `ml/features/pipeline.py`, `docs/feature_dictionary.md` — por qué las
  features de ventana necesitan historial.
- `ml/inference/predict.py`, `docs/m6_results.md` — el contrato ya
  implementado que este ADR documenta y confirma.
- Análisis de arquitectura inicial del proyecto — origen de la decisión "el
  servicio es stateless" que esta alternativa (a) mantiene.

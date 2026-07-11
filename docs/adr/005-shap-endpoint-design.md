# ADR-005: Diseño de endpoints — /predict rápido, /explain separado

## Estado
Aceptado

## Contexto

M5 midió el costo de calcular SHAP por predicción individual contra el costo de
solo predecir (`docs/m5_results.md`, sobre el modelo candidato, N=100 requests
simuladas):

| Métrica | Solo predicción | Predicción + SHAP |
|---|---|---|
| p50 | 4.53 ms | **14.20 ms** |
| p95 | 5.80 ms | **21.02 ms** |
| media | 4.68 ms | 14.75 ms |

Calcular SHAP agrega ~3.1x el costo de la predicción sola y representa ~76% del
tiempo total de una respuesta que incluyera ambas cosas. M5 dejó esto anotado
explícitamente como insumo cuantificado para esta decisión, sin resolverla.

Antes de escribir los routers de M7 hay que decidir la topología de endpoints:
¿un solo `/predict` con explicación opcional (parámetro/query), o dos endpoints
separados?

## Decisión

**Dos endpoints separados: `POST /predict` (sin SHAP, siempre rápido) y
`POST /explain` (predicción + explicación SHAP juntas, para la última lectura de
la ventana enviada).**

- `/predict` nunca calcula SHAP — el camino caliente por defecto paga solo
  ~4.5-5.8 ms (p50/p95), no ~14-21 ms.
- `/explain` recibe el mismo contrato de entrada que `/predict` (ver ADR-004:
  ventana de lecturas por pozo) y devuelve la predicción **más** el desglose SHAP
  de la última fila de la ventana (la lectura "actual") — no de toda la ventana,
  para no pagar N × ~14 ms en una sola llamada. Un cliente que ya llamó a
  `/predict` y después quiere entender el resultado hace una segunda llamada a
  `/explain` con la misma ventana; no hay una forma de pedir ambas cosas en una
  sola request en este diseño.
- Ambos endpoints comparten el mismo modelo cargado una vez al arrancar la app
  (`backend/app/main.py`, lifespan) — la diferencia de latencia es por el cálculo
  de SHAP en sí, no por recargar nada.

## Alternativas consideradas

1. **Un solo `/predict` con parámetro opcional `?explain=true`.** Un solo
   endpoint es más simple de integrar a primera vista, pero mezcla dos perfiles
   de latencia muy distintos (5ms vs. 21ms) bajo la misma ruta — más difícil de
   razonar sobre timeouts/rate limits/monitoreo por separado, y más fácil que un
   cliente termine pidiendo `explain=true` por accidente en un camino caliente
   (ej. un dashboard que hace polling cada pocos segundos) sin darse cuenta del
   costo. Descartada por eso, no porque técnicamente no funcionara.
2. **`/explain` explica toda la ventana recibida, no solo la última fila.** Más
   "completo", pero el costo escala linealmente con el tamaño de la ventana
   (una ventana de 20 lecturas costaría ~20×14ms ≈ 280ms) sin que quede claro
   que un cliente típico necesite explicar cada punto histórico, no solo la
   predicción actual. Descartada por sobre-pagar latencia para un caso de uso
   que no se pidió; se puede reconsiderar si aparece una necesidad real de
   explicar toda la ventana.
3. **Endpoint único que siempre calcula SHAP.** Simple, pero le clava ~76% de
   overhead a cada predicción sin excepción, incluyendo los casos (probablemente
   la mayoría) donde el consumidor solo quiere el número. Descartada — es
   exactamente el escenario que motivó medir la latencia en M5.

## Consecuencias

**Positivas**
- El camino por defecto (`/predict`) queda tan rápido como el modelo lo permite,
  sin que la decisión de explicabilidad le imponga costo a quien no la pide.
- Separar los endpoints hace explícito, en el propio diseño de la API, el
  trade-off que M5 cuantificó — no queda escondido en un parámetro opcional
  fácil de pasar por alto.
- Cada endpoint puede tener su propio timeout/límite de tasa si hace falta más
  adelante, sin afectar al otro.

**Negativas / riesgos**
- Un cliente que quiere predicción + explicación en el mismo momento hace 2
  requests, no 1 — hay algo de trabajo duplicado (la predicción se recalcula
  como parte de `/explain`, aunque sea barata comparada con el cálculo de SHAP).
- `/explain` solo cubre la última fila de la ventana — documentado como
  decisión explícita (alternativa 2), pero es una limitación real si en el
  futuro se necesita explicar un tramo completo, no un punto.

## Referencias

- `docs/m5_results.md` — medición de latencia que motiva esta decisión.
- `docs/adr/004-inference-input-contract.md` — contrato de entrada (ventana de
  lecturas) que ambos endpoints comparten.
- `ml/explainability/shap_explain.py` — la lógica de cálculo de SHAP que
  `backend/app/services` reutiliza, no reimplementa.

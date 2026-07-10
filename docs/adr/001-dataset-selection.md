# ADR-001: Selección del dataset de entrenamiento para predicción de ROP

## Estado
Aceptado — 2026-07-10

## Contexto

El objetivo del MVP es predecir Rate of Penetration (ROP) a partir de parámetros de
perforación (WOB, RPM, torque, hookload, flow rate, mud weight, profundidad). La
propuesta inicial consideraba tres fuentes: Volve (Equinor), WITSML públicos genéricos,
y Petrobras 3W.

Antes de comprometer el roadmap completo, se ejecutó un spike de verificación de datos
(2-3hs, descartable) para responder la pregunta que condicionaba todo el diseño
posterior: ¿el ROP está disponible como campo directo en los datos de Volve, o hay que
derivarlo de profundidad vs. tiempo?

## Decisión

Usar **USROP (University of Stavanger Rate of Penetration)**, un dataset derivado de
Volve, construido y publicado específicamente como benchmark académico para predicción
de ROP por Tunkiel, Sui y Wiktorski (2021), en lugar de:
- procesar archivos WITSML crudos de Volve desde cero, o
- derivar el ROP manualmente por diferenciación numérica de profundidad/tiempo.

## Detalles del dataset

| Campo | Valor |
|---|---|
| Nombre | USROP (University of Stavanger Rate of Penetration) |
| Fuente original | Volve Field dataset, Equinor (dominio público, 2018) |
| Pozos | 7 |
| Muestras | ~200,000 |
| Atributos | 12: MD, WOB, SPP, T (torque), RPM, FR (flow rate), DS (mud density), HD (hole diameter), HL (hookload), VD (vertical depth), GR, **ROP** |
| Convención de nombres | `USROP_A{revisión}_{id_pozo}.csv` (ej. `USROP_A 0 N-NA_F-9_Ad.csv`) |
| Repositorio | https://github.com/AndrzejTunkiel/USROP |
| Publicación de referencia | Tunkiel, A.T., Sui, D., Wiktorski, T. (2021). *Reference dataset for rate of penetration benchmarking*. Journal of Petroleum Science and Engineering, 196, 108069 |
| Licencia de los datos | CC BY-NC-SA 4.0 (No Comercial) |

## Alternativas consideradas y descartadas

1. **WITSML crudo de Volve sin preprocesar.** Requiere parsers propios, sin curaduría
   ni puntos de comparación externos. Alto costo de ingeniería para un beneficio que
   USROP ya resuelve.
2. **Derivar ROP manualmente de MD vs. tiempo.** Innecesario — el campo ya existe en
   USROP, evita ruido de derivación numérica sin ganancia real.
3. **Petrobras 3W.** Resuelve un problema distinto (detección de anomalías/eventos
   indeseados), no regresión de ROP. Se reserva para una fase posterior como módulo
   separado de detección de disfunciones.
4. **WITSML públicos genéricos (sin especificar fuente).** No existe tal dataset
   estandarizado; es un formato de intercambio, no una fuente de datos curada.

## Consecuencias

**Positivas**
- Resuelve de raíz el riesgo técnico #1 identificado en el análisis inicial (ambigüedad
  sobre disponibilidad de ROP como campo directo).
- Esquema de 12 atributos ya curado y consistente entre los 7 pozos — sin trabajo de
  homogeneización de nombres o unidades entre archivos.
- Existe un benchmark académico publicado contra el cual comparar resultados propios
  (MAE de referencia), lo cual da rigor verificable al portfolio, algo que el propio
  paper de origen señala como un problema extendido en la literatura de ROP con ML.
- La elección de MAE como métrica primaria (por el problema de ROP≈0 generando errores
  porcentuales infinitos) ya está justificada y documentada en la publicación de origen.

**Negativas / riesgos**
- Licencia de datos No Comercial (CC BY-NC-SA 4.0): debe documentarse por separado de
  la licencia del código del repositorio (MIT). El proyecto es "open source" en su
  código, no en los datos subyacentes.
- Dependencia de un repositorio mantenido por un investigador individual, no por una
  organización — riesgo de disponibilidad a largo plazo.
  **Mitigación**: script de descarga documentado (no versionar los CSV en el repo) +
  checksum de verificación; considerar un mirror propio solo si el original deja de
  estar disponible.

## Acciones derivadas (M1)

- [ ] Descargar el dataset USROP desde el repositorio oficial.
- [ ] Documentar el esquema completo (unidades, rangos, missing values por pozo) en
      `docs/data_dictionary.md`.
- [ ] Agregar sección de licenciamiento de datos al `README.md`, separada de la
      licencia del código.
- [ ] Fijar MAE como métrica primaria de evaluación (M8), documentando la razón
      (comportamiento cerca de ROP≈0) con cita a la publicación de origen.

## Referencias

- Tunkiel, A.T., Sui, D., Wiktorski, T. (2021). Reference dataset for rate of
  penetration benchmarking. *Journal of Petroleum Science and Engineering*, 196, 108069.
- Repositorio USROP: https://github.com/AndrzejTunkiel/USROP
- Equinor Volve data sharing: https://www.equinor.com/energy/volve-data-sharing

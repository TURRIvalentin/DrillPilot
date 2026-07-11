# Backlog

Ideas y tareas identificadas pero no implementadas, con el motivo por el que
quedaron afuera del alcance del milestone en el que surgieron. No es un roadmap
comprometido — es una lista de "no se hizo esto, y por qué" para no perder el
contexto ni repetir la misma investigación dos veces.

## Evaluar dataset completo de real-time drilling de Volve (Tunkiel)

**Origen:** M4, experimento de régimen (`docs/m4_results.md`) — el diagnóstico de
régimen mostró que el CV-pool solo tiene 2 pozos por régimen, insuficiente para
que un router de régimen (o cualquier modelo) aprenda la variabilidad real de
cada uno.

**Qué evaluar:** el dataset completo de perforación en tiempo real del campo
Volve, parseado de WITSML a CSV por Tunkiel (`ux.uis.no/~atunkiel`, ~2.7GB
comprimido) — USROP es un subconjunto curado de 7 pozos de este dataset más
grande; podría haber pozos adicionales, especialmente del régimen atípico
(somero/ROP alto), que ayuden a resolver el problema de escasez de pozos por
régimen identificado en M4.

**Por qué no se hizo:** bloqueado por acceso — `ux.uis.no` devuelve `403
Forbidden` a cualquier solicitud automatizada (confirmado con `WebFetch` y `curl`
con user-agent de navegador, ambos con el mismo resultado, así que es un bloqueo
del servidor, no de la herramienta). No se determinó cuántos pozos adicionales
existen con completitud comparable a los 12 atributos de USROP.

**Próximo paso, si se retoma:** revisión manual desde un navegador (posible que
el bloqueo sea por IP/región, no por tipo de cliente) para acceder a
`file_list.html` y contar pozos/completitud antes de comprometerse a descargar
los 2.7GB completos. Costo estimado: alto si hay que reprocesar WITSML/CSV de
pozos con completitud desigual — evaluar recién si el reconocimiento manual
confirma que vale la pena.

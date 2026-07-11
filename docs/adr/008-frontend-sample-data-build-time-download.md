# ADR-008: Datos de ejemplo del frontend -- descarga en build time, nunca commiteados

## Estado
Aceptado

## Contexto

El botón "cargar ejemplo real" del frontend (M8, `frontend/streamlit_app/sample_data.py`)
necesita filas reales de USROP para poblar el formulario -- llama a
`ml.features.dataset.load_combined_dataset()`, que lee todos los CSV de
`data/raw/` y falla con `FileNotFoundError` si el directorio está vacío
(`ml/features/dataset.py` línea 36).

Localmente, `docker-compose.yml` resuelve esto montando el host:
`./data:/app/data:ro`. Al desplegar a ECS Fargate (M11), no hay equivalente
de ese mount -- una task de Fargate no tiene volumen de host que montar, y
`frontend/Dockerfile` nunca copió los CSV a la imagen (no podría: `data/raw/`
está en `.gitignore` y no se commitea, ver ADR-001, licencia CC BY-NC-SA sin
redistribución). Resultado: en producción (ECS), el botón de ejemplo real
falla con `FileNotFoundError` -- bug encontrado durante la validación en
vivo de M11.

## Decisión

`frontend/Dockerfile` descarga y verifica el checksum de los CSV de USROP
**durante el build de la imagen**, corriendo el mismo script que ya usa el
setup de desarrollo local (`ml/data/download_usrop.py`, con
`docs/data_checksums.json` pinneado -- ver ADR-001). Los CSV quedan
horneados en la imagen final, en la misma ruta que
`ml.features.dataset.DEFAULT_RAW_DATA_DIR` ya espera (`/app/data/raw`
dentro del contenedor) -- ningún cambio de path en el código de la
aplicación.

Nunca se commitean al repositorio (`data/raw/` sigue en `.gitignore`) ni se
copian del host -- se obtienen de la misma fuente oficial
(`raw.githubusercontent.com/AndrzejTunkiel/USROP`) que ya usa el desarrollo
local, con la misma verificación de checksum que ya existe para detectar si
la fuente cambió de contenido sin aviso.

Es la misma lógica que ADR-006 punto 1 ya aplicó al backend (artefactos
horneados en build time, sin dependencia de red en runtime), extendida
ahora al dato de ejemplo del frontend -- no una decisión nueva, la
aplicación consistente de un principio ya aceptado a una parte del sistema
que quedó afuera quedó afuera hasta que M11 expuso el gap.

## Alternativas consideradas

### a) Descarga en runtime, al arrancar el contenedor

Descartada: mueve el punto de falla de "build time, visible de inmediato,
bloquea que una imagen rota llegue a pushearse" a "runtime, silencioso hasta
que un usuario aprieta el botón" -- exactamente el trade-off que ADR-006
punto 1 ya resolvió a favor de build time para el modelo del backend. Agrega
además latencia de arranque y un punto de falla de red en el contenedor
corriendo, no solo en el pipeline de build.

### b) Commitear los CSV al repositorio

Descartada de plano -- viola ADR-001 (CC BY-NC-SA, sin redistribución), ya
rechazada ahí explícitamente ("no versionar los CSV en el repo").

### c) Volumen EFS montado en la task de ECS

Descartada: infraestructura nueva (EFS, mount targets, permisos del task
role -- que hoy arranca sin ningún permiso adjunto, ADR-007 sección 1) para
resolver una función de conveniencia de demo, no una dependencia real del
servicio (`/predict` y `/explain` funcionan igual con datos ingresados a
mano, el botón de ejemplo real es un atajo, no un requisito).

## Consecuencias

**Positivas**
- El frontend queda autocontenido y correcto en ECS, con la misma propiedad
  de "sin dependencia de red externa en runtime" que ya tiene el backend.
- Un build con la fuente de USROP inalcanzable o con contenido cambiado
  falla el build (o el job de CI) de inmediato -- no llega a producción una
  imagen que falla silenciosamente para el usuario final.
- Mismo comportamiento entre local, Docker Compose y ECS -- no hay una
  ruta de código nueva ni un caso especial para producción.

**Negativas / riesgos**
- El build de la imagen de frontend ahora hace una llamada de red real a
  GitHub durante CI (jobs `docker-build` y `deploy`) -- una dependencia de
  red nueva en el pipeline de build que antes no existía (antes CI solo
  tocaba PyPI/uv y, en `deploy`, AWS). Si `raw.githubusercontent.com` no
  está disponible, **el build del frontend falla** -- consecuencia directa
  y aceptada de esta decisión, no un caso borde no contemplado. Se acepta
  explícitamente dado el alcance del proyecto (portfolio, sin SLA de
  disponibilidad de build ni de despliegue continuo) -- si esto fuera un
  servicio con requisitos de build siempre-verde, la mitigación sería un
  mirror propio del dataset (ya anotado como opción futura en ADR-001) en
  vez de depender del repositorio de un investigador individual en cada
  build.
- Imagen de frontend más pesada (~33 MB de CSV horneados, ver
  `data/raw/` local). Sin impacto en el backend -- no toca esos datos.

## Referencias

- `docs/adr/001-dataset-selection.md` -- licencia CC BY-NC-SA y la decisión
  de no versionar los CSV, que esta ADR respeta.
- `docs/adr/006-model-packaging-deploy.md` sección 1 -- el principio de
  "artefactos horneados en build time, no dependencias externas en
  runtime" que esta ADR extiende al frontend.
- `ml/data/download_usrop.py` -- el script de descarga con verificación de
  checksum que `frontend/Dockerfile` ahora corre en build time.
- `frontend/streamlit_app/sample_data.py` -- el consumidor de estos datos
  (botón "cargar ejemplo real").

# ADR-009: `FileNotFoundError` persistente -- dos copias físicas de `ml/` en la imagen, `__file__` resuelve distinto según el entry point

## Estado
Aceptado -- opción (a), modo editable, implementada en ambos Dockerfiles
(`frontend/Dockerfile` y el `Dockerfile` raíz del backend) y validada. Ver
"Decisión final" y "Validación" al final.

## Contexto

Después de ADR-008 (CSVs horneados en build time), el botón "cargar ejemplo
real" del frontend en ECS seguía fallando con `FileNotFoundError` al
recargar la página. La hipótesis inicial (mismatch de rutas entre build y
runtime) y la siguiente (imagen real de ECR distinta a la reconstrucción
local) se descartaron con evidencia -- la imagen real, pulleada por digest
exacto y corrida con su `CMD` real, tenía los CSV correctamente en
`/app/data/raw/` (ver conversación de M11, validación en vivo).

La causa real, confirmada con evidencia directa contra la imagen real:

**`uv pip install --system ".[ml,frontend]"` (sin `-e`) instala una segunda
copia física completa de `ml/` en site-packages, además de la que ya vive
en `/app/ml/` por el `COPY ml ./ml` del Dockerfile.** Confirmado con
`find / -name "clean_usrop.py"` dentro de la imagen real:

```
/root/.cache/uv/archive-v0/PjJeObP83xsM9J17/ml/cleaning/clean_usrop.py
/usr/local/lib/python3.12/site-packages/ml/cleaning/clean_usrop.py
/app/build/lib/ml/cleaning/clean_usrop.py
/app/ml/cleaning/clean_usrop.py
```

`ml/cleaning/clean_usrop.py` calcula la ruta de datos así:

```python
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DATA_DIR = _REPO_ROOT / "data" / "raw"
```

Esto resuelve a una ruta **distinta según cuál copia física importa
Python** -- y eso depende de cómo se invoca el proceso, no de qué copia
"es la correcta":

- Un `python -c "import ml; print(ml.__file__)"` (probado primero, por
  error metodológico) agrega el directorio de trabajo actual (`/app`) a
  `sys.path` -- encuentra `/app/ml/__init__.py` primero. **Este test dio
  un falso negativo**, no representativo del proceso real.
- El `CMD` real del Dockerfile (`streamlit run
  frontend/streamlit_app/app.py`) es un entry point de consola: Python
  **no** agrega el cwd a `sys.path` en ese caso. Streamlit inserta
  explícitamente el directorio del script (`bootstrap._fix_sys_path`,
  `/app/frontend/streamlit_app`) -- que no contiene `ml/` -- y la
  resolución de `import ml` cae a site-packages.

Confirmado replicando fielmente esa invocación (`python <script>` en el
mismo directorio que `app.py`, no `python -c`) contra la imagen real
(digest `sha256:1e2399580a...`, tag `3a4b133`):

```
sys.path[0]: /app/frontend/streamlit_app
ml.__file__: /usr/local/lib/python3.12/site-packages/ml/__init__.py
DEFAULT_RAW_DATA_DIR: /usr/local/lib/python3.12/site-packages/data/raw
exists: False
```

Ahí está el `FileNotFoundError`: la copia de `ml` que realmente se importa
en producción calcula una ruta de datos dentro de `site-packages`, que no
existe -- los CSV sí existen, en `/app/data/raw/`, pero esa no es la copia
que Streamlit termina cargando.

## Decisión pendiente

### a) Instalar en modo editable (`uv pip install --system -e ".[ml,frontend]"`)

Site-packages deja de contener una copia física de `ml/` -- en su lugar,
un mecanismo de enlace (PEP 660) apunta de vuelta a `/app/ml`. Una sola
fuente de verdad física, sin importar qué entry point importe el paquete.

- **A favor:** arregla la causa raíz de una vez -- no solo este caso de uso
  (`DEFAULT_RAW_DATA_DIR`), sino cualquier otro código presente o futuro en
  `ml/`, `backend/` o `frontend/` que use `Path(__file__)` para ubicarse a
  sí mismo. Cambio de una palabra en el Dockerfile (`-e`), no toca ningún
  `.py`. Elimina además la copia duplicada de `~30 MB extra en la imagen
  (`site-packages/ml/` + `/app/build/lib/ml/`, este último un artefacto de
  build de setuptools que tampoco debería estar ahí).
- **En contra:** depende de que el árbol fuente (`/app/ml`, etc.) siga
  presente en la imagen final -- cierto hoy (`frontend/Dockerfile` es
  single-stage), pero si este patrón se copiara a un build multi-stage
  como el del backend (`Dockerfile` raíz, que arma el venv en un stage
  `builder` y copia solo `/opt/venv` al stage final), el enlace editable
  apuntaría a una ruta que podría no existir en el stage final salvo que
  también se recopie el código fuente ahí (que de hecho ya se hace,
  `COPY ml ./ml` en el stage `backend` final -- pero es una coincidencia
  de que ambos stages usan el mismo `WORKDIR /app`, no una garantía
  estructural). No es un problema para el frontend hoy, pero merece una
  nota si se extiende el patrón.

### b) Variable de entorno configurable para la ruta de datos

Reemplazar `Path(__file__).resolve().parents[2] / "data" / "raw"` por una
variable de entorno con default explícito (ej.
`DRILLPILOT_RAW_DATA_DIR`, default = el cálculo actual como fallback para
no romper desarrollo local/tests), y fijarla en `frontend/Dockerfile` con
`ENV DRILLPILOT_RAW_DATA_DIR=/app/data/raw`.

- **A favor:** no depende de entender cómo cada entry point arma
  `sys.path` -- la ruta queda explícita y configurable por entorno, un
  patrón más "12-factor". Más fácil de razonar para alguien sin el
  contexto de esta investigación.
- **En contra:** arregla el síntoma en `ml/cleaning/clean_usrop.py`
  (y por herencia `ml/features/dataset.py`), pero dos copias físicas de
  `ml/` siguen existiendo en la imagen -- cualquier otro módulo que use
  `Path(__file__)` de forma similar (hoy no hay otro caso, pero nada lo
  impide en el futuro) tiene el mismo bug latente, sin que este ADR lo
  prevenga. Requiere tocar código (`clean_usrop.py`, posiblemente
  `download_usrop.py` por consistencia) más el Dockerfile, no un cambio
  de una palabra.

## Recomendación

**(a), modo editable.** Es un cambio de menor superficie (una palabra en
`frontend/Dockerfile`, cero archivos `.py` tocados) que resuelve la causa
raíz -- la existencia de dos copias físicas divergentes -- en vez de
parchar el único síntoma ya encontrado. Dado que este proyecto ya tiene el
precedente de preferir entender y resolver la causa raíz sobre el síntoma
puntual (ver, por ejemplo, cómo se investigó este mismo bug en vez de
asumir la primera hipótesis), (a) es más consistente con esa práctica. La
única reserva real (build multi-stage) no aplica al frontend hoy.

## Decisión final

Se implementó **(a)** en los dos Dockerfiles que hacen `uv pip install
".[...]"` de este proyecto -- ambos usaban el mismo patrón no-editable:

- `frontend/Dockerfile`: `RUN uv pip install --system -e ".[ml,frontend]"`.
- `Dockerfile` (raíz, imagen de backend, stage `builder`):
  `RUN uv pip install --python /opt/venv/bin/python -e ".[ml,api]"`.

El backend se incluyó como prevención: su cadena de imports en runtime
(`backend/app/services/inference_service.py` -> `ml.evaluation.metrics`,
`ml.features.pipeline`, `ml.inference.predict`) no toca hoy ningún módulo
que use `Path(__file__)` para ubicarse (confirmado con
`grep -rn "__file__" backend/ ml/` -- 8 archivos en total, ninguno
importado por el backend en runtime), así que no había síntoma visible ahí
-- pero es el mismo patrón de install no-editable, mismo bug latente si
algún import futuro lo alcanza.

La reserva sobre build multi-stage (sección "a) Instalar en modo
editable", "En contra") se probó directamente contra el backend real
(`Dockerfile` raíz, builder + stage final) y **no se materializó**: ambos
stages usan el mismo `WORKDIR /app` y el stage final vuelve a copiar
`ml/`, así que el enlace editable creado en `builder` sigue apuntando a
una ruta que existe en la imagen final. Confirmado en la validación de
abajo.

## Validación

**Frontend** (misma metodología que encontró el bug: `CMD` real, script
copiado junto a `app.py` y corrido como `python <script>`, no `python -c`):

```
sys.path[0]: /app/frontend/streamlit_app
ml.__file__: /app/ml/__init__.py
```
```
DEFAULT_RAW_DATA_DIR: /app/data/raw
exists: True
```
```
rows: 198928
wells: [0, 1, 2, 3, 4, 5, 6]
sample window rows: 10
```

`find / -name "clean_usrop.py"` dentro de la imagen con el fix: una sola
copia, `/app/ml/cleaning/clean_usrop.py` -- ya no hay copia en
`site-packages` ni en `/app/build/lib`.

Contenedor `healthy` (`HEALTHCHECK` del Dockerfile), `/_stcore/health` ->
200.

**Backend** (regresión, ya que se tocó preventivamente sin que hubiera
síntoma):

```
ml.__file__: /app/ml/__init__.py
```
```json
{"status":"ok","model_loaded":true,"model_uri":"/app/docker/model_artifact"}
```

`/predict` contra la misma ventana de 10 filas usada en la validación en
vivo de M11 devolvió predicciones idénticas a las de antes del cambio --
sin regresión.

## Referencias

- `ml/cleaning/clean_usrop.py` -- `_REPO_ROOT`/`DEFAULT_RAW_DATA_DIR`, el
  cálculo cuyo resultado depende de qué copia física se importa.
- `frontend/Dockerfile` -- `RUN uv pip install --system ".[ml,frontend]"`,
  el install no-editable que crea la segunda copia.
- `docs/adr/008-frontend-sample-data-build-time-download.md` -- el fix
  anterior (CSVs horneados en build time), correcto en sí mismo pero
  insuficiente por este bug de resolución de import, descubierto después.
- Streamlit `bootstrap._fix_sys_path` -- inserta el directorio del script
  principal en `sys.path`, no el cwd; el comportamiento que hace que el
  `CMD` real del Dockerfile no encuentre `/app/ml` por el mecanismo que sí
  lo encuentra un `python -c` interactivo.

# ADR-006: Empaquetado del modelo y target de despliegue

## Estado
Aceptado

## Contexto

Hasta M8, el backend carga el modelo desde el Model Registry de MLflow en
runtime (`ml/inference/predict.py::MODEL_URI = "models:/drillpilot-rop@production"`,
resuelto contra el tracking store local en cada arranque). Antes de escribir el
Dockerfile hay que decidir cómo el modelo llega a producción: ¿la imagen carga
desde un servidor de tracking en runtime, o el artefacto viaja empaquetado dentro
de la imagen?

Un hallazgo durante esta decisión, relevante para el diseño: el tracking store
local de este proyecto (`sqlite:///.../mlflow.db`) vive, por una particularidad de
cómo MLflow arma la URI en Windows con espacios en la ruta, en una carpeta
*hermana* con nombre percent-encoded literal
(`AI%20Engineer%20Portfolio/DrillPilot/mlflow.db`, no
`AI Engineer Portfolio/DrillPilot/mlflow.db` con espacios reales) -- fuera del
propio repositorio. Esto no afecta el desarrollo local (el cliente de MLflow
resuelve la URI de forma consistente y reproducible, verificado con múltiples
corridas), pero confirma que depender del tracking store local dentro del build de
Docker sería frágil y no portable a otra máquina/CI. Es una razón adicional,
concreta, para la decisión de este ADR, no solo una preferencia arquitectónica
abstracta.

Además, se verificó directamente que copiar tal cual el directorio de artefacto
interno de MLflow (`mlruns/1/models/m-<id>/artifacts/`) y cargarlo con
`mlflow.sklearn.load_model(<path llano>)` **falla** en Windows
(`MlflowException: No such artifact`) -- la función interna de MLflow que separa
"raíz" de "artifact_path" a partir de una ruta local no reconstruye ese layout
correctamente. En cambio, `mlflow.sklearn.save_model(model, out_dir)` seguido de
`mlflow.sklearn.load_model(out_dir)` sí funciona de punta a punta sin tocar ningún
tracking store, verificado con las mismas predicciones exactas que ya reportó
producción (ver `docs/m9_m10_results.md`). Esta es la técnica concreta que la
decisión de este ADR usa para producir el artefacto que la imagen empaqueta.

## Decisión

### 1. El artefacto se empaqueta dentro de la imagen en build time

La imagen Docker **no** contacta a MLflow (ni un tracking server, ni el store
local) en runtime. El pipeline combinado (features + LightGBM, el candidato
promovido a "production" en M6) se exporta **una vez**, fuera de Docker, a un
directorio plano y autocontenido
(`ml/inference/export_model.py`, ver sección 4) que se commitea al repositorio en
`docker/model_artifact/` y que el Dockerfile copia con un `COPY` plano -- sin
ninguna dependencia de MLflow durante el build ni el runtime del contenedor más
allá de poder deserializar el pickle (`mlflow.sklearn.load_model` sobre una
carpeta local, sin red).

**Justificación:** para un MVP donde el modelo no se reentrena en vivo, mantener
un servidor de tracking persistente y alcanzable desde producción solo para
resolver un alias en cada arranque es infraestructura que no se necesita todavía
-- agrega una dependencia de red externa (latencia, disponibilidad, autenticación)
a algo que puede resolverse una vez, en build time, y quedar fijo. El artefacto
queda versionado junto con la imagen: dado un tag de imagen, siempre se sabe
exactamente qué modelo corre adentro.

### 2. Consecuencia explícita: no hay hot-swap del modelo

Si el modelo se reentrena, **no hay manera de actualizarlo sin reconstruir y
redesplegar la imagen**. No existe (todavía) un mecanismo de recarga en caliente
ni de apuntar el contenedor corriendo a un nuevo artefacto sin reiniciarlo. Esto
es una limitación conocida y aceptada para el MVP, documentada acá explícitamente
para que no quede implícita: cualquier ciclo de reentrenamiento futuro
**incluye** re-exportar el artefacto (`ml/inference/export_model.py`),
reconstruir la imagen y redesplegar -- no es una tarea aparte que se pueda
olvidar.

### 3. Target de despliegue: AWS ECS Fargate

Se elige **ECS Fargate** (contenedor, sin gestionar servidores) sobre **Lambda**
(containerizado) para el deploy real (fuera de scope de este milestone, pero la
decisión de empaquetado depende de saber el target).

**Justificación:** las dependencias pesadas del backend (`lightgbm`, `shap`,
`mlflow` -- este último solo para deserializar el pickle en runtime, no para
tracking) arrastran binarios nativos y un footprint de imagen considerable.
Lambda impone un límite de 10 GB para imágenes de contenedor y, más importante en
la práctica, un cold start notablemente peor cuanto más pesada es la imagen y más
grande el proceso de inicialización de un proceso Python con estas librerías
cargadas en memoria -- justo el perfil de este backend. Fargate no tiene ese
límite de tamaño de imagen ni el mismo penalty de cold start (el proceso queda
corriendo continuamente, no se re-inicializa por invocación), a costa de no
escalar a cero entre requests -- un trade-off aceptable para un servicio que se
espera con tráfico sostenido, no esporádico.

### 4. Referencia explícita y reproducible de la versión del modelo

`ml/inference/export_model.py` expone `PINNED_PRODUCTION_RUN_ID`, el `run_id` de
MLflow de la corrida `promote-to-production` (M6) que registró
`drillpilot-rop` v1 (`a302711c1cd34d4da4c86235387dc5f8`) -- **no** el alias
`"production"` resuelto al momento de exportar. El script:

1. Carga el modelo desde `runs:/{run_id}/model` (requiere el tracking store
   local -- se corre en la máquina/entorno de desarrollo, nunca dentro del build
   de Docker ni en el contenedor).
2. Lo re-guarda con `mlflow.sklearn.save_model(..., serialization_format="cloudpickle")`
   en un directorio plano y portable (`docker/model_artifact/` por default).

El `run_id` es un argumento explícito (`--run-id`, default
`PINNED_PRODUCTION_RUN_ID`) -- correr el script sin pensarlo exporta el candidato
ya pinneado en el código, nunca "lo que sea que esté aliaseado production ahora
mismo" de forma silenciosa. Cambiar de modelo es un cambio de una línea
(`PINNED_PRODUCTION_RUN_ID`) más volver a correr el export, no una resolución
implícita en build time.

El Dockerfile referencia esto en un comentario (ver `Dockerfile`) y el
`docker/model_artifact/` exportado se commitea al repositorio -- la alternativa
de buscarlo en un artifact store remoto durante `docker build` o el workflow de CI
queda deliberadamente fuera de este milestone (ver sección "Alternativas").

## Alternativas consideradas

### a) Cargar desde el Model Registry en runtime (statu quo hasta M8)

Es lo que hace `ml/inference/predict.py` hoy vía `models:/drillpilot-rop@production`.
**Descartada para producción** -- requiere un tracking server alcanzable desde el
contenedor en todo momento, agrega latencia/punto de falla de red al arranque, y
"production" es un alias mutable: dos builds de la misma imagen en momentos
distintos podrían resolver modelos distintos sin que quede registrado en ningún
lado cuál. Se mantiene disponible como mecanismo (vía la variable de entorno
`DRILLPILOT_MODEL_URI`, ver `backend/app/core/config.py`) para desarrollo local
contra un tracking store real, pero no es lo que la imagen de producción usa.

### b) Descargar el artefacto de un artifact store remoto (S3/registry) durante el build de Docker o el workflow de CI

Más "correcto" a largo plazo (no requiere commitear binarios al repo), pero
implica credenciales, un bucket/registry ya aprovisionado y IaC para gestionarlo
-- exactamente el trabajo que el enunciado de este milestone excluye
explícitamente ("Terraform/IaC, credenciales, redes... queda para un milestone
posterior"). Se pospone a cuando el deploy real a ECS Fargate se implemente;
hasta entonces, commitear el artefacto exportado (~2 MB, ver
`docs/m9_m10_results.md`) a `docker/model_artifact/` es reproducible sin
infraestructura nueva.

### c) Copiar el directorio de artefacto interno de MLflow tal cual (`mlruns/.../artifacts/`) en el `COPY` del Dockerfile

Descartada por la razón técnica de la sección Contexto: falla al cargar con
`mlflow.sklearn.load_model()` sobre una ruta llana en Windows. Independientemente
de si funcionara, tampoco sería la opción más limpia -- arrastraría archivos de
metadata de MLflow (`conda.yaml`, `registered_model_meta`) pensados para
interactuar con un tracking store, no para ejecución standalone.

## Consecuencias

**Positivas**
- El contenedor de producción no tiene ninguna dependencia de red hacia MLflow --
  arranca y sirve predicciones completamente offline respecto del tracking store.
- Reproducible: dado el mismo `docker/model_artifact/` commiteado, cualquier
  build de la imagen (local o en CI) empaqueta exactamente el mismo modelo.
- La referencia al modelo (`PINNED_PRODUCTION_RUN_ID`) queda en el código,
  versionada junto con el resto del proyecto -- no en un valor implícito
  resuelto en build time.

**Negativas / riesgos**
- Sin hot-swap (sección 2) -- limitación conocida y aceptada, no oculta.
- Commitear un artefacto binario al repositorio es una desviación de la práctica
  habitual de no versionar binarios grandes en git; aceptable acá por el tamaño
  (~2 MB) y porque evita introducir infraestructura de artifact store antes de
  que el deploy real la necesite (ver alternativa b). Si el modelo creciera
  significativamente de tamaño, esta decisión debería revisarse.
- `ml/inference/export_model.py` sigue dependiendo del tracking store local para
  el **export** (paso 1, fuera de Docker) -- la ADR no elimina esa dependencia,
  solo la saca de runtime/build de la imagen. Correrlo requiere el entorno de
  desarrollo con el tracking store poblado (`mlruns/` + `mlflow.db`).

## Referencias

- `ml/training/promote_model.py` (M6) -- construye y registra el pipeline
  combinado que este ADR empaqueta.
- `ml/inference/export_model.py` -- implementa la exportación descripta acá.
- `docs/m9_m10_results.md` -- verificación del round-trip
  `save_model`/`load_model` y de las predicciones reales contra el contenedor.
- `docs/adr/004-inference-input-contract.md` -- el servicio permanece stateless;
  este ADR no lo modifica, solo cambia de dónde sale el modelo que ese servicio
  usa.

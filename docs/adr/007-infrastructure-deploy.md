# ADR-007: Arquitectura de infraestructura y despliegue

## Estado
Propuesto -- pendiente de revisión del usuario. No se aplicó ningún recurso de
AWS ni se escribió código de Terraform todavía; este documento es puramente de
diseño, previo a esa implementación (ver "Próximos pasos" al final).

## Contexto

ADR-006 fijó el target de despliegue (ECS Fargate) y cómo se empaqueta el
modelo (horneado en la imagen, sin tracking server en runtime). Falta decidir
la infraestructura AWS concreta que sostiene ese despliegue, cómo se
provisiona con Terraform de forma reproducible, y cómo se controla el costo
para un proyecto de portfolio que no tiene tráfico 24/7 real.

Cuatro problemas de "arranque" (bootstrapping) aparecen antes de poder escribir
Terraform "normal":

- Terraform necesita un backend remoto (S3 + lock) para que el state no viva
  solo en una laptop -- pero ese backend es en sí mismo infraestructura que
  Terraform tendría que crear, sin poder todavía guardar su propio state ahí
  (no existe el bucket hasta después de crearlo).
- El workflow de CI/CD (M10) necesita autenticarse contra AWS para aplicar
  Terraform -- la forma correcta es OIDC (sin credenciales estáticas en
  GitHub Secrets), pero el proveedor OIDC y el rol de IAM que la CI va a
  asumir son ellos mismos recursos de Terraform: no pueden crearse usando el
  rol que todavía no existe.

Ambos son el mismo patrón: un primer `apply` manual, acotado y documentado,
que "arranca" la cadena de confianza antes de que el resto sea 100%
autogestionado por CI.

## Decisión

### 1. Recursos de AWS para ECS Fargate

| Recurso | Cardinalidad | Notas |
|---|---|---|
| ECR | 2 repos (`drillpilot-backend`, `drillpilot-frontend`) | Tags inmutables por SHA de commit -- mismo principio de referencia explícita y reproducible que ADR-006 punto 4 (nunca `:latest` como referencia de despliegue). |
| Cluster ECS | 1, Fargate (sin EC2 capacity provider) | No hay servidores propios que gestionar. |
| Task definition + service | 1 par por servicio (backend, frontend) | Refleja la misma topología de dos contenedores que `docker-compose.yml` ya usa en local. |
| IAM: task execution role | 1, compartido | Rol que ECS usa para arrancar el contenedor: pull de ECR + escritura en CloudWatch Logs. Política administrada `AmazonECSTaskExecutionRolePolicy` alcanza -- es exactamente el permiso mínimo que este rol necesita, no hay razón para escribir una custom. |
| IAM: task role | 1 por servicio, mínimo privilegio | El rol que asume la *aplicación* en runtime. Hoy ninguno de los dos servicios llama a una API de AWS directamente (el modelo está horneado en la imagen, no en S3; no hay Secrets Manager todavía) -- el task role arranca **sin ningún permiso adjunto**. Se le agregan permisos puntuales el día que haya una necesidad real (ej. leer un secret), no de forma preventiva. |
| Security groups | 1 para las tasks | Inbound en el puerto de la app (8000 backend, 8501 frontend) desde el origen que la sección siguiente determina; outbound abierto (necesario para el pull de ECR y llamadas entre servicios). |
| CloudWatch Logs | 1 log group por servicio | **Con retención explícita** (ej. 14 días) -- el default de "nunca expira" acumula costo de almacenamiento indefinidamente en un proyecto que no necesita retener logs de demos viejas. |

### 2. Application Load Balancer vs. IP pública directa

**Decisión: IP pública directa en la task de Fargate (`assignPublicIp: ENABLED`
en una subnet pública), no ALB, para este alcance.**

| | ALB | IP pública directa |
|---|---|---|
| Costo fijo | ~USD 16-25/mes de base (por-hora) + LCUs, **corre aunque el servicio ECS esté en 0 tasks** -- es un recurso separado, no lo apaga el runbook de la sección 4. | Sin costo fijo propio; AWS cobra la IPv4 pública por hora (~USD 0.005/h, ~USD 3.6/mes) **solo mientras la task está corriendo** -- si el servicio está en 0 tasks, ese costo también es 0. |
| HTTPS | Terminación TLS con certificado de ACM -- URL con candado real. | No hay TLS nativo; HTTP plano o habría que agregar un proxy/túnel aparte (fuera de alcance ahora). |
| Estabilidad de la URL | DNS name del ALB, estable entre despliegues. | La IP pública cambia cada vez que Fargate reemplaza la task (nuevo ENI en cada arranque) -- hay que consultarla de nuevo antes de cada demo (un solo comando de AWS CLI, documentado en el runbook). |
| Representa un despliegue real de empresa | Sí -- balanceo, health checks propios, blue/green, WAF. | Menos -- válido para 1 task fija, no escala el patrón a N tasks sin agregar el ALB después. |
| Complejidad de Terraform | Listener, target group, certificado ACM, SG adicional. | Solo el SG de la task. |

**Razón de la recomendación:** el patrón de uso real de este proyecto es
"demos ocasionales", no tráfico sostenido -- exactamente el escenario que la
sección 4 (control de costo) quiere resolver escalando a 0 tasks entre usos.
Un ALB rompe esa estrategia a medias: el compute se apaga, pero el ALB sigue
facturando las 24 horas del día independientemente. Para que "apagar el
servicio" signifique realmente "costo casi cero", la IP pública directa es la
opción consistente con el resto del diseño. La pérdida de HTTPS y de URL
estable es real y se documenta como limitación aceptada -- no oculta -- y
migrar a ALB más adelante (si el proyecto necesita un dominio propio, HTTPS,
o más de una task) es un cambio localizado, no una reescritura.

### 3. Bootstrap de OIDC (autenticación de CI/CD sin credenciales estáticas)

El workflow de GitHub Actions (M10) debe poder aplicar Terraform y hacer
`docker push` a ECR sin guardar una access key de AWS de larga duración en
GitHub Secrets. El mecanismo correcto es OIDC: GitHub emite un token firmado
por request, y un rol de IAM configurado con un *trust policy* que confía en
el proveedor OIDC de GitHub lo intercambia por credenciales temporales.

El problema: el proveedor OIDC y ese rol de IAM **son recursos de Terraform**,
y Terraform necesita credenciales de AWS para crear cualquier cosa, incluido
el mecanismo que le daría credenciales sin usar una key estática. No se puede
resolver este ciclo automatizándolo dentro del propio pipeline que depende de
él.

**Decisión:** el primer `terraform apply` que crea el proveedor OIDC + el rol
de IAM para CI se corre **manualmente, una sola vez**, con las credenciales de
un usuario de IAM de arranque (`drillpilot-bootstrap` o similar):

- Creado a mano en la consola o CLI de AWS por el operador humano (el
  usuario) -- **nunca la cuenta root**, y con MFA habilitado.
- Permisos acotados a lo que el bootstrap necesita crear: IAM (proveedor
  OIDC + roles + políticas), S3 y DynamoDB (para el backend de state, sección
  4). No `AdministratorAccess` de forma permanente.
- Sus credenciales (access key) se generan, se usan una vez para el `apply`
  de bootstrap, y se **desactivan o eliminan inmediatamente después** -- una
  vez que el rol OIDC existe, ningún flujo posterior (CI ni el operador)
  vuelve a necesitarlas. No se guardan en ningún secret de GitHub.
- Todo `apply` posterior (tanto de infraestructura como de despliegues de
  aplicación) pasa por el workflow de GitHub Actions asumiendo el rol via
  OIDC -- cero credenciales estáticas de AWS viviendo en el repositorio o en
  GitHub Secrets desde ese punto en adelante.

Este paso queda documentado acá como manual y explícito, no como un
`TODO` implícito que alguien podría intentar automatizar por error más
adelante.

### 4. Backend remoto de Terraform state

Bucket S3 (versionado, cifrado, acceso bloqueado a público) + tabla DynamoDB
(lock, previene que dos `apply` concurrentes corrompan el state) --
creados **antes** del stack principal, en un bootstrap separado.

Mismo problema de arranque que la sección 3: el bucket que va a alojar el
state de Terraform no puede ser, él mismo, gestionado por un Terraform que ya
esté usando ese bucket como backend (no existe todavía en el primer
`terraform init`). Se resuelve así:

- El bootstrap (bucket S3 + tabla DynamoDB + proveedor OIDC + rol de IAM de
  CI, sección 3 -- **un solo `apply` chico**, no dos separados, corridos con
  las mismas credenciales del usuario de arranque) usa **state local** (el
  archivo `terraform.tfstate` en la máquina del operador, no versionado en
  git -- son ~4 recursos, de bajísima frecuencia de cambio).
- El stack principal (cluster, servicios, IAM de las tasks, etc.) sí usa el
  backend S3+DynamoDB recién creado desde su primer `init`.
- El state local del bootstrap se resguarda (backup fuera del repo, ej. un
  gestor de secretos personal) -- si se pierde, el bootstrap es
  reproducible corriendo el mismo `apply` de nuevo (es idempotente y de bajo
  riesgo), no una fuente de verdad operativa continua como el state del
  stack principal.

### 5. Por qué `terraform/` no usa módulos todavía

**Decisión: un único root module (archivos `.tf` planos, organizados por
tema -- `ecs.tf`, `iam.tf`, `networking.tf`, `ecr.tf`, `variables.tf`,
`outputs.tf` -- pero sin ningún `module "..." { source = ... }`).**

Los módulos de Terraform pagan su costo de abstracción cuando hay múltiples
instancias del mismo conjunto de recursos para parametrizar (varios
ambientes -- dev/staging/prod --, o el mismo componente reusado en varios
proyectos). Ninguna de las dos condiciones aplica acá: un solo ambiente, un
cluster, dos servicios que no se van a duplicar. Envolver esto en módulos
hoy sería la misma sobre-ingeniería que el resto del proyecto evita
deliberadamente (ver la regla general del proyecto de no introducir
abstracciones más allá de lo que el alcance actual requiere) -- una capa de
indirección sin un segundo caso de uso que la justifique. Si en el futuro
aparece un segundo ambiente o el patrón se reusa en otro proyecto, extraer
módulos en ese momento es un refactor acotado, no algo que haya que
prever de antemano.

## Alternativas consideradas

- **ECS con EC2 (capacity provider) en vez de Fargate:** descartado ya en
  ADR-006 -- gestionar instancias EC2 es exactamente la carga operativa que
  Fargate evita, sin beneficio para el tamaño de este servicio.
- **CodePipeline/CodeBuild de AWS en vez de GitHub Actions:** descartado --
  el proyecto ya vive en GitHub con un workflow de CI funcionando (M10);
  agregar un segundo sistema de CI/CD solo para el deploy fragmenta el
  pipeline sin necesidad.
- **Terraform Cloud/HCP para el state en vez de S3+DynamoDB:** válido y
  más manejado, pero agrega una cuenta/servicio externo más para un
  proyecto de portfolio de un solo colaborador -- S3+DynamoDB alcanza y ya
  vive dentro de la misma cuenta de AWS que el resto.
- **Credenciales estáticas de un usuario de IAM para todo el CI (sin
  OIDC):** más simple de configurar, pero es exactamente la práctica que
  OIDC existe para evitar (una access key de larga duración con permisos de
  deploy, guardada en un secret, que no expira ni rota sola). Se usa un
  usuario con key estática *solo* para el bootstrap de una sola vez
  (sección 3), nunca para el flujo continuo de CI.

## Consecuencias

**Positivas**
- Costo alineado con el uso real: apagar el servicio (runbook,
  `docs/runbook_ecs_scale_down.md`) efectivamente lleva el costo operativo a
  ~0, porque no hay un ALB corriendo de fondo.
- Cadena de confianza de CI sin credenciales estáticas de larga duración
  después del bootstrap -- superficie de ataque mínima si GitHub Secrets se
  filtrara.
- El bucket de state y el proveedor OIDC quedan aislados del stack
  principal -- destruir/recrear el stack principal (ej. para iterar el
  diseño) nunca arriesga el backend de state ni la identidad de CI.

**Negativas / riesgos**
- Sin HTTPS ni URL estable (sección 2) -- limitación aceptada para el MVP,
  documentada, no un descuido.
- El bootstrap (secciones 3 y 4) es un paso manual fuera del pipeline
  automatizado -- exige que un humano lo corra correctamente una vez, con
  sus propias credenciales, y limpie la key después. Es un punto de fallo
  humano que Terraform normal no tiene, inherente a resolver el problema del
  huevo y la gallina.
- El state local del bootstrap (sección 4) no tiene el mismo nivel de
  protección (versionado, lock) que el state del stack principal -- riesgo
  aceptado dado su bajísima frecuencia de cambio (se toca una vez, casi
  nunca de nuevo).
- Sin ALB, escalar a más de una task no tiene un balanceador esperándolas --
  no es un problema hoy (1 task fija), pero es explícitamente lo primero que
  habría que agregar antes de escalar horizontalmente.

## Referencias

- `docs/adr/006-model-packaging-deploy.md` -- fija el target ECS Fargate y
  el empaquetado del modelo que esta infraestructura sirve.
- `docs/runbook_ecs_scale_down.md` -- el procedimiento operativo de apagado
  mencionado en la sección 4.
- `.github/workflows/ci.yml` (M10) -- el workflow que, tras el bootstrap de
  la sección 3, va a asumir el rol de IAM via OIDC para desplegar.

## Próximos pasos (no incluidos en este ADR)

1. El usuario configura una alerta de presupuesto (AWS Budgets) y el
   usuario de IAM de bootstrap (sección 3) -- fuera del alcance de este
   documento, y de lo que Terraform gestiona (el usuario de bootstrap no
   puede crearse a sí mismo con Terraform por la misma razón circular de la
   sección 3).
2. Recién con eso confirmado, se escribe el código de Terraform
   (`terraform/`) que implementa las decisiones de este ADR -- todavía no
   existe ni se corrió `terraform apply` contra ningún recurso real.

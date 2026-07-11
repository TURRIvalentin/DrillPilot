# Terraform main: ECR, ECS Fargate, IAM/OIDC

Stack principal de infraestructura (ver `docs/adr/007-infrastructure-deploy.md`).
Usa el backend remoto S3+DynamoDB creado en `terraform/bootstrap/` (ver el
README de esa carpeta) -- state en S3 desde el primer `init`, no local.

## Proveedor OIDC de GitHub Actions: referenciado, no gestionado acá

Al escribir este stack se encontró que la cuenta de AWS ya tiene un
proveedor OIDC para `token.actions.githubusercontent.com` (creado antes de
este stack, fuera de Terraform). Para no colisionar con él -- AWS permite
un solo proveedor OIDC por URL por cuenta, así que crearlo de nuevo como
`resource "aws_iam_openid_connect_provider"` haría fallar el `apply` con
`EntityAlreadyExists` -- este stack lo **referencia** con un data source
(`data "aws_iam_openid_connect_provider" "github_actions"` en `iam.tf`) en
vez de crearlo o importarlo. El rol de CI confía en su ARN, pero este
stack nunca lo crea, actualiza ni destruye.

## Bootstrap del rol de CI: el primer apply es manual

Igual que el backend de state, el rol de IAM que GitHub Actions asume vía
OIDC (`iam.tf`) es en sí mismo un recurso de Terraform, y el workflow de CI
lo necesita para poder aplicar/desplegar -- no puede crearse a sí mismo. El
primer `apply` de este stack se corre manualmente, con el profile de AWS
CLI `drillipilot-bootstrap` (ver ADR-007 sección 3). Después de ese primer
apply, el ARN del rol (output `github_actions_role_arn`) se usa en el
workflow de GitHub Actions (`permissions: id-token: write` + el step de
`aws-actions/configure-aws-credentials` con `role-to-assume`).

## Por qué los servicios de ECS arrancan en 0 tasks

`backend_desired_count` y `frontend_desired_count` (`variables.tf`) tienen
default `0`. En el primer apply todavía no hay ninguna imagen real
pusheada a los repos de ECR (los task definitions usan el tag `bootstrap`,
que no existe) -- desired_count en 0 evita que ECS intente arrancar tasks
que fallarían el pull de imagen. Prender los servicios (después de que CI
haga el primer push real) es manual, vía
`docs/runbook_ecs_scale_down.md`, no vía `terraform apply`.

## Por qué `task_definition` y `desired_count` están en `ignore_changes`

CI actualiza la task definition del servicio en cada deploy
(`ecs:RegisterTaskDefinition` + `ecs:UpdateService`, ver el rol en
`iam.tf`) y el runbook cambia `desired_count` a mano entre demos. Sin
`ignore_changes` en `aws_ecs_service`, un `terraform apply` posterior de
este stack (por ejemplo, para agregar un recurso no relacionado)
revertiría ambos a los valores fijos de este código -- deshaciendo el
último deploy de CI o volviendo a apagar/prender el servicio sin querer.

## Comandos

```bash
cd terraform/main
terraform init
terraform plan
terraform apply   # requiere confirmación manual explícita, ver regla acordada con el usuario
```

# Terraform bootstrap: backend remoto de state

Este stack crea el backend remoto (S3 + DynamoDB) que el stack principal de
Terraform (`terraform/`, todavía no creado -- ver M11 parte 2) va a usar para
guardar y lockear su state. Ver `docs/adr/007-infrastructure-deploy.md`
sección 4 para el razonamiento completo.

## Por qué este stack usa state local

El bucket que va a alojar el state remoto no puede, él mismo, estar
gestionado por un Terraform que ya use ese bucket como backend -- no existe
todavía en el primer `init`. Por eso este stack (bootstrap) usa el backend
por defecto (`local`, archivo `terraform.tfstate` en esta carpeta) y **no**
se versiona en git (ver `.gitignore` en la raíz del repo).

Son ~5 recursos de bajísima frecuencia de cambio (se tocan una vez, casi
nunca de nuevo). Si el state local se pierde, el bootstrap es reproducible
corriendo `terraform apply` de nuevo -- es idempotente. Igualmente, hacé un
backup del archivo `terraform.tfstate` de esta carpeta fuera del repo (ej.
un gestor de secretos personal) después de aplicar.

## Recursos que crea

| Recurso | Nombre | Notas |
|---|---|---|
| Bucket S3 | `drillipilot-tfstate-<account_id>` | Versionado, cifrado SSE-AES256, acceso público bloqueado (las 4 flags de `aws_s3_bucket_public_access_block`). |
| Tabla DynamoDB | `drillipilot-tfstate-lock` | `PAY_PER_REQUEST`, clave de partición `LockID` (String) -- formato que requiere el backend `s3` de Terraform para locking. |

El sufijo `<account_id>` evita colisión con el namespace global de nombres
de bucket S3 (es único por cuenta de AWS).

## Cómo usarlo desde el stack principal

Terraform >= 1.9 (misma versión mínima que este bootstrap declara en
`versions.tf`).

Una vez que este bootstrap esté aplicado, el stack principal (`terraform/`,
M11 parte 2) declara este bloque en su propio `versions.tf` (o archivo
equivalente) **antes de su primer `terraform init`**:

```hcl
terraform {
  backend "s3" {
    bucket         = "drillipilot-tfstate-750907156542"
    key            = "drillpilot/terraform.tfstate"
    region         = "us-east-2"
    dynamodb_table = "drillipilot-tfstate-lock"
    encrypt        = true
  }
}
```

- `bucket` / `dynamodb_table`: los nombres exactos de arriba (o los outputs
  `tfstate_bucket_name` / `tfstate_lock_table_name` de este stack, si se
  corre `terraform output` acá primero).
- `key`: ruta dentro del bucket para el state de ese stack -- separada de
  cualquier otro state que eventualmente viva en el mismo bucket.
- `region`: `us-east-2`, la misma región donde se creó el bucket.
- `encrypt = true`: cifra el state en tránsito/reposo además del cifrado a
  nivel de bucket ya configurado acá.

Después de agregar el bloque, correr `terraform init` en el stack principal
migra (o inicializa directamente) el state hacia S3.

## Comandos

```bash
cd terraform/bootstrap
terraform init
terraform plan
terraform apply   # requiere confirmación manual, ver ADR-007 sección 3
```

Corrido con el profile de AWS CLI `drillipilot-bootstrap` (`AWS_PROFILE` o
`-profile` en el provider, región `us-east-2`) -- credenciales de un usuario
de IAM de arranque, no la cuenta root. Ver ADR-007 sección 3 sobre el ciclo
de vida de esas credenciales.

variable "aws_region" {
  description = "Región de AWS donde se despliega el stack principal."
  type        = string
  default     = "us-east-2"
}

variable "aws_profile" {
  description = "Profile de AWS CLI para este apply (mismo usuario de bootstrap, ver ADR-007 sección 3)."
  type        = string
  default     = "drillipilot-bootstrap"
}

variable "name_prefix" {
  description = "Prefijo para nombrar los recursos de este stack (coincide con los nombres ya usados en docs/runbook_ecs_scale_down.md)."
  type        = string
  default     = "drillpilot"
}

variable "github_repo" {
  description = "Repo de GitHub (owner/name) autorizado a asumir el rol de CI vía OIDC, acotado a la rama main."
  type        = string
  default     = "TURRIvalentin/DrillPilot"
}

variable "image_tag" {
  description = "Tag de imagen usado en la task definition del apply inicial -- todavía no hay ninguna imagen real pusheada a ECR. CI registra nuevas revisiones con el tag real (SHA de commit) en cada deploy; ver el lifecycle.ignore_changes de aws_ecs_service en ecs.tf."
  type        = string
  default     = "bootstrap"
}

variable "backend_desired_count" {
  description = "Cantidad deseada de tasks del backend. 0 en el apply inicial -- no hay imagen real en ECR todavía. Se prende a mano (docs/runbook_ecs_scale_down.md) o vía CI, nunca vía Terraform (ver ignore_changes)."
  type        = number
  default     = 0
}

variable "frontend_desired_count" {
  description = "Cantidad deseada de tasks del frontend. Mismo razonamiento que backend_desired_count."
  type        = number
  default     = 0
}

variable "backend_cpu" {
  description = "CPU units de Fargate para el backend (512 = 0.5 vCPU)."
  type        = number
  default     = 512
}

variable "backend_memory" {
  description = "Memoria (MB) de Fargate para el backend -- combinación válida con backend_cpu."
  type        = number
  default     = 1024
}

variable "frontend_cpu" {
  description = "CPU units de Fargate para el frontend (256 = 0.25 vCPU) -- Streamlit es liviano, solo llama al backend."
  type        = number
  default     = 256
}

variable "frontend_memory" {
  description = "Memoria (MB) de Fargate para el frontend -- combinación válida con frontend_cpu."
  type        = number
  default     = 512
}

variable "log_retention_days" {
  description = "Retención de CloudWatch Logs (ADR-007 sección 1 -- evita acumular costo de almacenamiento indefinidamente)."
  type        = number
  default     = 14
}

variable "ecr_max_image_count" {
  description = "Cantidad máxima de imágenes que retiene cada repo ECR antes de expirar las más viejas (ADR-007 sección 1)."
  type        = number
  default     = 10
}

variable "frontend_backend_url" {
  description = "URL pública del backend que usa el frontend (env var DRILLPILOT_BACKEND_URL). Sin ALB ni service discovery (ADR-007 sección 2), la IP pública del backend cambia en cada arranque de task -- este valor se actualiza a mano después de levantar el backend (docs/runbook_ecs_scale_down.md), no hay forma de resolverlo automáticamente con este diseño."
  type        = string
  default     = "http://backend-ip-pending:8000"
}

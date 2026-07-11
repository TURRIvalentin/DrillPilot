variable "aws_region" {
  description = "Región de AWS donde se crea el backend remoto de Terraform."
  type        = string
  default     = "us-east-2"
}

variable "aws_profile" {
  description = "Profile de AWS CLI usado para el apply de bootstrap (credenciales de una sola vez, ver ADR-007 sección 3)."
  type        = string
  default     = "drillipilot-bootstrap"
}

variable "name_prefix" {
  description = "Prefijo para nombrar los recursos de este stack."
  type        = string
  default     = "drillipilot"
}

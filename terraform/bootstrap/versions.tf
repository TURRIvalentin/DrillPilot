terraform {
  required_version = ">= 1.9"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Bootstrap del backend remoto: todavía no existe el bucket/tabla que
  # alojarían el state, así que este stack usa state local a propósito.
  # Ver README.md de esta carpeta.
}

provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile

  default_tags {
    tags = {
      Project   = "drillpilot"
      Component = "terraform-bootstrap"
      ManagedBy = "terraform"
    }
  }
}

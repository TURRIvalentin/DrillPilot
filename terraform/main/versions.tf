terraform {
  required_version = ">= 1.9"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Backend remoto creado en terraform/bootstrap/ (ver ese README.md) --
  # bucket S3 versionado/cifrado/privado + tabla DynamoDB para el lock.
  backend "s3" {
    bucket         = "drillipilot-tfstate-750907156542"
    key            = "drillpilot/terraform.tfstate"
    region         = "us-east-2"
    dynamodb_table = "drillipilot-tfstate-lock"
    encrypt        = true
  }
}

provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile

  default_tags {
    tags = {
      Project   = "drillpilot"
      Component = "terraform-main"
      ManagedBy = "terraform"
    }
  }
}

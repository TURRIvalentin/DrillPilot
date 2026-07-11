# Sin VPC propia: se usa el VPC default de la cuenta (ya tiene subnets
# públicas con ruta a un Internet Gateway), consistente con el resto del
# diseño de no construir infraestructura que este proyecto no necesita.
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Un solo security group para ambas tasks (ADR-007 sección 1): abre
# únicamente el puerto de cada app, sin ALB de por medio -- IP pública
# directa (ADR-007 sección 2).
resource "aws_security_group" "tasks" {
  name        = "${var.name_prefix}-tasks"
  description = "DrillPilot ECS Fargate tasks -- direct public IP, no ALB (ADR-007)"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "Backend API (FastAPI: /predict, /explain, /health)"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Frontend UI (Streamlit)"
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "Outbound open -- required for ECR image pull and calls between services (ADR-007 section 1)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

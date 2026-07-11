output "ecr_backend_repository_url" {
  description = "URL del repo ECR del backend (para docker push / docker tag en CI)."
  value       = aws_ecr_repository.backend.repository_url
}

output "ecr_frontend_repository_url" {
  description = "URL del repo ECR del frontend."
  value       = aws_ecr_repository.frontend.repository_url
}

output "ecs_cluster_name" {
  description = "Nombre del cluster ECS (usado por docs/runbook_ecs_scale_down.md)."
  value       = aws_ecs_cluster.this.name
}

output "ecs_backend_service_name" {
  description = "Nombre del servicio ECS del backend."
  value       = aws_ecs_service.backend.name
}

output "ecs_frontend_service_name" {
  description = "Nombre del servicio ECS del frontend."
  value       = aws_ecs_service.frontend.name
}

output "github_actions_role_arn" {
  description = "ARN a usar en el workflow de GitHub Actions (permissions: id-token: write, y el role-to-assume del step de configure-aws-credentials)."
  value       = aws_iam_role.github_actions_deploy.arn
}

output "tfstate_bucket_name" {
  description = "Nombre del bucket S3 para el state remoto del stack principal."
  value       = aws_s3_bucket.tfstate.id
}

output "tfstate_bucket_arn" {
  description = "ARN del bucket S3 para el state remoto."
  value       = aws_s3_bucket.tfstate.arn
}

output "tfstate_lock_table_name" {
  description = "Nombre de la tabla DynamoDB usada para el lock de state."
  value       = aws_dynamodb_table.tfstate_lock.name
}

output "aws_region" {
  description = "Región donde se crearon los recursos de bootstrap."
  value       = var.aws_region
}

output "aws_account_id" {
  description = "Account ID de AWS usado para el sufijo del bucket."
  value       = data.aws_caller_identity.current.account_id
}

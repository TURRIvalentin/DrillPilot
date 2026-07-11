data "aws_caller_identity" "current" {}

# --- Proveedor OIDC de GitHub Actions -------------------------------------
# Ya existe en la cuenta (creado fuera de este stack, antes de este apply) --
# se referencia, no se gestiona acá. Ver terraform/main/README.md.
data "aws_iam_openid_connect_provider" "github_actions" {
  url = "https://token.actions.githubusercontent.com"
}

# --- Rol que GitHub Actions asume vía OIDC --------------------------------

data "aws_iam_policy_document" "github_actions_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [data.aws_iam_openid_connect_provider.github_actions.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    # Least privilege también en la condición de confianza, no solo en los
    # permisos: acotado exactamente a la rama main de este repo. Ninguna
    # otra rama (ni PRs de forks, ni otras ramas del mismo repo) ni ningún
    # otro repo puede asumir este rol.
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repo}:ref:refs/heads/main"]
    }
  }
}

resource "aws_iam_role" "github_actions_deploy" {
  name               = "${var.name_prefix}-github-actions-deploy"
  assume_role_policy = data.aws_iam_policy_document.github_actions_trust.json
}

data "aws_iam_policy_document" "github_actions_deploy" {
  # Obtener el token de login de Docker contra ECR (equivalente a
  # `aws ecr get-login-password`). GetAuthorizationToken no soporta scoping
  # a nivel de recurso -- es un requisito de la API de AWS, no una elección
  # de este diseño; Resource = "*" es inevitable acá.
  statement {
    sid       = "ECRAuth"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  # Subir capas de imagen y el manifest final (equivalente a `docker push`),
  # acotado a los dos repos de este proyecto -- no a cualquier repo de ECR
  # de la cuenta.
  statement {
    sid    = "ECRPush"
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "ecr:PutImage",
      "ecr:InitiateLayerUpload",
      "ecr:UploadLayerPart",
      "ecr:CompleteLayerUpload",
    ]
    resources = [
      aws_ecr_repository.backend.arn,
      aws_ecr_repository.frontend.arn,
    ]
  }

  # Registrar una nueva revisión de task definition con la imagen recién
  # pusheada. RegisterTaskDefinition tampoco soporta scoping a nivel de
  # recurso -- mismo requisito de la API que GetAuthorizationToken.
  statement {
    sid       = "ECSRegisterTaskDefinition"
    effect    = "Allow"
    actions   = ["ecs:RegisterTaskDefinition"]
    resources = ["*"]
  }

  # Leer la task definition actual de cada familia (por nombre de familia,
  # sin revisión -- la forma estándar de pedir "la revisión activa"). Igual
  # que RegisterTaskDefinition, DescribeTaskDefinition no soporta permisos a
  # nivel de recurso: la guía de IAM de ECS la lista sin "Resource types"
  # scopeable, y en la práctica AWS evalúa el request contra Resource "*"
  # incluso con un ARN acotado en la policy (confirmado en CI: la llamada
  # con family name devolvió AccessDeniedException "on resource: *" pese al
  # ARN scoped que había acá antes) -- Resource = "*" es inevitable acá, no
  # un permiso amplio "por las dudas".
  statement {
    sid       = "ECSDescribeTaskDefinition"
    effect    = "Allow"
    actions   = ["ecs:DescribeTaskDefinition"]
    resources = ["*"]
  }

  # Apuntar el servicio corriendo a la nueva revisión de task definition,
  # acotado a los dos servicios de este proyecto -- no a cualquier servicio
  # del cluster.
  statement {
    sid    = "ECSUpdateService"
    effect = "Allow"
    actions = [
      "ecs:UpdateService",
      "ecs:DescribeServices",
    ]
    resources = [
      aws_ecs_service.backend.id,
      aws_ecs_service.frontend.id,
    ]
  }

  # RegisterTaskDefinition necesita poder "pasarle" el execution role y los
  # task roles a ECS -- acotado exactamente a los 3 roles de este proyecto,
  # y solo cuando el servicio que los recibe es ECS (PassedToService), nunca
  # para pasarlos a otro servicio de AWS.
  statement {
    sid     = "PassRolesToECS"
    effect  = "Allow"
    actions = ["iam:PassRole"]
    resources = [
      aws_iam_role.ecs_task_execution.arn,
      aws_iam_role.backend_task.arn,
      aws_iam_role.frontend_task.arn,
    ]

    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "github_actions_deploy" {
  name   = "${var.name_prefix}-deploy"
  role   = aws_iam_role.github_actions_deploy.id
  policy = data.aws_iam_policy_document.github_actions_deploy.json
}

# --- Rol de ejecución de ECS (compartido) ---------------------------------
# Rol que ECS usa para arrancar el contenedor: pull de ECR + escritura en
# CloudWatch Logs. La política administrada AmazonECSTaskExecutionRolePolicy
# alcanza -- es exactamente el permiso mínimo que este rol necesita, no hay
# razón para escribir una custom (ADR-007 sección 1).

data "aws_iam_policy_document" "ecs_tasks_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ecs_task_execution" {
  name               = "${var.name_prefix}-ecs-task-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_trust.json
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# --- Roles de task por servicio (sin permisos todavía) --------------------
# El rol que asume la aplicación en runtime. Ninguno de los dos servicios
# llama a una API de AWS directamente hoy (el modelo está horneado en la
# imagen, no en S3; no hay Secrets Manager todavía) -- arrancan sin ningún
# permiso adjunto. Se agregan permisos puntuales el día que haya una
# necesidad real, no de forma preventiva (ADR-007 sección 1).

resource "aws_iam_role" "backend_task" {
  name               = "${var.name_prefix}-backend-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_trust.json
}

resource "aws_iam_role" "frontend_task" {
  name               = "${var.name_prefix}-frontend-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_trust.json
}

# Runbook: apagar/prender el servicio de ECS entre demos

Este proyecto es un portfolio, no un servicio 24/7 -- el objetivo de este
runbook es que el medidor de AWS no corra sin necesidad entre una demo y la
siguiente. Ver [docs/adr/007-infrastructure-deploy.md](adr/007-infrastructure-deploy.md)
para por qué esto realmente lleva el costo a ~0 (IP pública directa, sin ALB
corriendo de fondo) en vez de solo reducirlo parcialmente.

**Nota:** este runbook asume que `terraform/` ya existe y el servicio de ECS
ya fue desplegado -- todavía no es el caso (ver ADR-007, "Próximos pasos").
Se documenta ahora para no dejarlo pendiente una vez que sí exista.

## Apagar (después de una demo)

Escala el servicio a 0 tasks -- ECS detiene la task corriendo, se libera la
IP pública asociada y deja de facturarse compute:

```bash
aws ecs update-service \
  --cluster drillpilot \
  --service <nombre-del-servicio> \
  --desired-count 0
```

Repetir para cada servicio (`drillpilot-backend`, `drillpilot-frontend`) si
corren como servicios separados.

Verificar que efectivamente bajó:

```bash
aws ecs describe-services \
  --cluster drillpilot \
  --services <nombre-del-servicio> \
  --query "services[0].{deseado:desiredCount,corriendo:runningCount}"
```

## Prender (antes de una demo)

```bash
aws ecs update-service \
  --cluster drillpilot \
  --service <nombre-del-servicio> \
  --desired-count 1
```

La task tarda entre ~30 segundos y un par de minutos en arrancar (pull de la
imagen desde ECR + `HEALTHCHECK` del contenedor, ver `Dockerfile`) antes de
quedar `RUNNING` y pasar el health check.

## Conseguir la IP pública actual

Sin ALB (ver ADR-007 sección 2), la IP pública es distinta cada vez que la
task arranca -- no hay una URL fija para compartir de antemano. Conseguirla
después de prender el servicio:

```bash
TASK_ARN=$(aws ecs list-tasks --cluster drillpilot --service-name <nombre-del-servicio> --query "taskArns[0]" --output text)
ENI_ID=$(aws ecs describe-tasks --cluster drillpilot --tasks "$TASK_ARN" \
  --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" --output text)
aws ec2 describe-network-interfaces --network-interface-ids "$ENI_ID" \
  --query "NetworkInterfaces[0].Association.PublicIp" --output text
```

## Verificación rápida de que el servicio realmente responde

```bash
curl http://<ip-publica>:8000/health   # backend
curl -o /dev/null -w "%{http_code}\n" http://<ip-publica>:8501   # frontend
```

`/health` debe reportar `"model_loaded": true` -- igual que en local (M7) y
en Docker (M9), no alcanza con que el puerto responda.

## Costo si te olvidás de apagarlo

Con IP pública directa y sin ALB, dejar 1 task de cada servicio corriendo sin
usar cuesta el compute de Fargate (vCPU/memoria por hora) más la IP pública
(~USD 0.005/hora cada una) -- no hay un costo fijo grande escondido (a
diferencia de un ALB, que factura aunque no haya tasks corriendo). Igual,
apagar entre demos es gratis y toma un comando -- no hay razón para dejarlo
prendido de más.

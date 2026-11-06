#!/usr/bin/env bash
# Prepara la cuenta de AWS una sola vez, desde un PC con la AWS CLI y permisos de administrador:
# bucket protegido, token de GitHub en SSM y el rol IAM de la instancia.
#
#   BUCKET=simpsons-frinkiac-123 REGION=us-east-1 bash scripts/newDataset/frinkiac/ec2/preparar_aws.sh
#
# El token de GitHub (fine-grained, solo este repo, permiso "Issues: Read and write") se pide por
# teclado y nunca queda en el historial de la shell.
set -euo pipefail

BUCKET="${BUCKET:?Define BUCKET}"
REGION="${REGION:?Define REGION}"
PARAMETRO="${PARAMETRO:-/frinkiac/github-token}"
ROL="${ROL:-frinkiac-ec2}"
CARPETA="$(cd "$(dirname "$0")" && pwd)"
CUENTA="$(aws sts get-caller-identity --query Account --output text)"

echo "== Bucket s3://$BUCKET ($REGION)"
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  if [ "$REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION"
  else
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
      --create-bucket-configuration LocationConstraint="$REGION"
  fi
fi
# Versionado: ninguna sobrescritura o borrado pierde la versión anterior
aws s3api put-bucket-versioning --bucket "$BUCKET" --versioning-configuration Status=Enabled
aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-encryption --bucket "$BUCKET" --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
aws s3api put-bucket-lifecycle-configuration --bucket "$BUCKET" --lifecycle-configuration '{
  "Rules": [{"ID": "versiones-antiguas", "Status": "Enabled", "Filter": {},
             "NoncurrentVersionExpiration": {"NoncurrentDays": 30},
             "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7}}]}'

echo "== Token de GitHub en SSM ($PARAMETRO)"
read -rsp "Pega el token de GitHub (no se muestra): " TOKEN; echo
aws ssm put-parameter --region "$REGION" --name "$PARAMETRO" --type SecureString --overwrite \
  --value "$TOKEN" >/dev/null
unset TOKEN

echo "== Rol IAM $ROL"
if ! aws iam get-role --role-name "$ROL" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROL" --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}' >/dev/null
fi
POLITICA="$(sed -e "s/BUCKET/$BUCKET/g" -e "s/REGION/$REGION/g" -e "s/CUENTA/$CUENTA/g" \
  -e "s#parameter/frinkiac/github-token#parameter${PARAMETRO}#" "$CARPETA/iam_policy.json")"
aws iam put-role-policy --role-name "$ROL" --policy-name frinkiac --policy-document "$POLITICA"
if ! aws iam get-instance-profile --instance-profile-name "$ROL" >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$ROL" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$ROL" --role-name "$ROL"
fi

echo "Listo. Lanza la instancia con el perfil '$ROL' y ec2/setup_ec2.sh como user-data."

#!/usr/bin/env bash
# Instala el servicio de recolección en una EC2 con Amazon Linux 2023 (t3.small basta; spot sirve).
# Se usa como user-data al lanzar la instancia, o con "sudo bash setup_ec2.sh" ya dentro de ella.
# Es idempotente: volver a ejecutarlo actualiza el código y reinicia el servicio.
#
# Edita estas variables antes de usarlo como user-data:
BUCKET="${BUCKET:-CAMBIAR-nombre-del-bucket}"
REGION="${REGION:-us-east-1}"
PARAMETRO="${PARAMETRO:-/frinkiac/github-token}"
REPO_URL="${REPO_URL:-https://github.com/ByFlopp/LosSimpsons_DeepLearning.git}"
RAMA="${RAMA:-main}"
set -euo pipefail

if [[ "$BUCKET" == CAMBIAR* ]]; then
  echo "Define BUCKET antes de ejecutar setup_ec2.sh" >&2
  exit 1
fi

dnf install -y git
PYTHON=""
for version in 3.12 3.11; do
  if dnf install -y "python$version" "python$version-pip"; then
    PYTHON="python$version"
    break
  fi
done
[ -n "$PYTHON" ] || { echo "No se pudo instalar Python 3.11 o 3.12" >&2; exit 1; }

id frinkiac >/dev/null 2>&1 || useradd --system --home-dir /var/lib/frinkiac --shell /sbin/nologin frinkiac

mkdir -p /opt/frinkiac
if [ -d /opt/frinkiac/repo/.git ]; then
  git -C /opt/frinkiac/repo fetch --depth 1 origin "$RAMA"
  git -C /opt/frinkiac/repo reset --hard "origin/$RAMA"
else
  git clone --depth 1 --branch "$RAMA" "$REPO_URL" /opt/frinkiac/repo
fi
[ -d /opt/frinkiac/venv ] || "$PYTHON" -m venv /opt/frinkiac/venv
/opt/frinkiac/venv/bin/pip install --upgrade pip
/opt/frinkiac/venv/bin/pip install -r /opt/frinkiac/repo/scripts/newDataset/frinkiac/requirements-frinkiac.txt

cat > /etc/frinkiac.env <<EOF
FRINKIAC_BUCKET=$BUCKET
AWS_REGION=$REGION
AWS_DEFAULT_REGION=$REGION
FRINKIAC_SSM_TOKEN=$PARAMETRO
FRINKIAC_DATOS=/var/lib/frinkiac
EOF
chmod 0640 /etc/frinkiac.env
chgrp frinkiac /etc/frinkiac.env

install -m 0644 /opt/frinkiac/repo/scripts/newDataset/frinkiac/ec2/frinkiac.service /etc/systemd/system/frinkiac.service
systemctl daemon-reload
systemctl enable frinkiac
systemctl restart frinkiac
echo "Servicio iniciado. Registros: journalctl -u frinkiac -f"

#!/usr/bin/env bash
# Arranca la demo a pantalla completa. Uso: bash run_mac.sh
# Los modelos se cambian en vivo con las teclas 1-4, no hace falta elegir aquí.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "[X] Falta el entorno. Ejecuta primero:  bash setup_mac.sh"
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if ! ls checkpoints/*/latest_net_G.pth >/dev/null 2>&1; then
  echo "[X] No hay ningún modelo en checkpoints/."
  exit 1
fi

echo "=== Learning to See ==="
echo
echo "  [1] Fuego    papel arrugado"
echo "  [2] Mar      trapo o camisa"
echo "  [3] Nubes    papel arrugado"
echo "  [4] Flores   papel arrugado simulando un petalo"
echo
echo "  [h] textos   [+/-] umbrales   [s] captura   [q] salir"
echo
echo "La primera vez macOS pedirá permiso de cámara."
echo

python webcam_inference.py --half "$@"

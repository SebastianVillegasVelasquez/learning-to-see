#!/usr/bin/env bash
# Instalación en macOS (Apple Silicon: M1 a M5).
# Uso:  bash setup_mac.sh
set -euo pipefail

cd "$(dirname "$0")"

echo "=== Learning to See — instalación para macOS ==="
echo

# --- 1) Comprobar arquitectura ---
ARCH="$(uname -m)"
if [ "$ARCH" != "arm64" ]; then
  echo "[!] Arquitectura detectada: $ARCH (no arm64)."
  echo "    En Macs Intel no hay aceleración MPS; funcionará en CPU (pocos fps)."
  echo
fi

# --- 2) Comprobar Python ---
if ! command -v python3 >/dev/null 2>&1; then
  echo "[X] No se encontró python3."
  echo "    Instálalo con:  brew install python@3.11"
  echo "    (o desde https://www.python.org/downloads/macos/)"
  exit 1
fi
PYV="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "Python detectado: $PYV"
case "$PYV" in
  3.9|3.10|3.11|3.12|3.13) ;;
  *) echo "[!] Versión no probada. Se recomienda 3.11." ;;
esac

# --- 3) Entorno virtual ---
if [ ! -d ".venv" ]; then
  echo "Creando entorno virtual en .venv ..."
  python3 -m venv .venv
else
  echo "Entorno .venv ya existe, se reutiliza."
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel >/dev/null

# --- 4) Dependencias ---
# En macOS la build estándar de PyTorch YA incluye MPS: no hay que pasar ningún
# índice extra (a diferencia de Windows/Linux, que necesitan el índice de CUDA).
echo "Instalando dependencias (puede tardar unos minutos)..."
python -m pip install \
  "torch>=2.4.0" \
  "torchvision>=0.19.0" \
  "numpy>=1.24.3" \
  "opencv-python>=4.9.0" \
  "Pillow>=10.0.0" \
  "dominate>=2.8.0" \
  "tqdm>=4.66.0"

# --- 5) Clonar pix2pix si falta (solo se usa models/networks.py) ---
if [ ! -f "pix2pix/models/networks.py" ]; then
  echo "Clonando pytorch-CycleGAN-and-pix2pix en ./pix2pix ..."
  git clone --depth 1 https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git pix2pix
else
  echo "pix2pix ya está presente."
fi

# --- 6) Verificación ---
echo
echo "=== Verificación ==="
python - <<'PY'
import platform
import torch
print("torch          :", torch.__version__)
print("plataforma     :", platform.platform())
b = getattr(torch.backends, "mps", None)
ok = bool(b and b.is_available() and b.is_built())
print("MPS disponible :", "SI" if ok else "NO")
if ok:
    x = torch.randn(256, 256, device="mps")
    y = (x @ x).sum().item()
    torch.mps.synchronize()
    print("prueba en GPU  : ok")
else:
    print("  -> se usará CPU (pocos fps)")
PY

echo
echo "=== Listo ==="
echo "Para arrancar:"
echo "  source .venv/bin/activate"
echo "  python webcam_inference.py --checkpoint ./checkpoints/fire/latest_net_G.pth --t1 50 --t2 150"
echo
echo "O simplemente:  bash run_mac.sh"
echo
echo "NOTA: la primera vez macOS pedirá permiso de cámara para tu terminal."
echo "      Si no aparece: Ajustes del Sistema > Privacidad y seguridad > Cámara."

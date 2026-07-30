# Learning to See — réplica técnica (pix2pix / océano)

Réplica de la arquitectura de **"Learning to See"** de Memo Akten: un pix2pix entrenado
sobre **un solo dominio** de imágenes (aquí: océano/olas), donde el par de entrenamiento no
son dos imágenes distintas, sino **la misma imagen y su versión de bordes**:

```
A (input)  = cv2.Canny(imagen)      ← lo que el modelo "ve"
B (target) = la imagen real         ← lo que el modelo "imagina"
```

Al inferir, cualquier entrada de bordes (una webcam, un trapo, una mano) se traduce al único
dominio que el modelo conoce. Ese es el punto conceptual de la pieza.

Base de código: [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
(equivalente moderno del `affinelayer/pix2pix-tensorflow` en TF1 que usó Akten).
Patrón de preprocesado tomado de [memoakten/webcam-pix2pix-tensorflow](https://github.com/memoakten/webcam-pix2pix-tensorflow)
y [joshmurr/cci-auto-pix2pix](https://github.com/joshmurr/cci-auto-pix2pix).

---

## Modelos entrenados

| Modelo | Dataset | Imágenes | Canny | Épocas | L1 val |
|---|---|---|---|---|---|
| **Fuego** | `entrenamiento/fire_dataset` | 682 | t=50/150 | 100 | **17,3**/255 |
| **Mar** | `entrenamiento/sea_dataset` | 522 (filtrado) | t=20/60, blur 5, median 3 | 130 | 36,3/255 |
| Flores rojas | `entrenamiento/imagenes_flores` | 197 | t=40/120 | 200 | 26,2/255 |

Para inferencia, ver [README_INFERENCE.md](README_INFERENCE.md). Los umbrales se aplican
solos según el modelo elegido.

## Estructura

```
learning_to_see/
├── pix2pix/                 # clon de pytorch-CycleGAN-and-pix2pix (train.py, test.py, models/…)
├── data/
│   ├── raw_images/          # ← TUS imágenes de océano aquí (jpg/png). No se versiona.
│   ├── AB/
│   │   ├── train/           # pares A|B de 512x256 generados por preprocess_canny.py
│   │   └── val/             # opcional (--val_frac)
│   └── preview/             # salida del modo --preview, para calibrar umbrales
├── preprocess_canny.py      # raw_images → pares A|B
├── check_dataset.py         # valida raw_images antes de preprocesar/subir
├── requirements.txt
├── .venv/                   # entorno virtual local (no se versiona)
└── .gitignore
```

> El entrenamiento **no** se hace aquí: el entorno local es CPU-only y sirve para preparar
> el dataset. El entrenamiento va en Google Colab con GPU.

---

## 1. Entorno

Ya creado en `.venv` (Python 3.11, torch CPU + OpenCV). Para activarlo:

```powershell
.\.venv\Scripts\Activate.ps1      # PowerShell
```
```bash
source .venv/Scripts/activate     # Git Bash
```

Recrearlo desde cero:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## 2. Validar las imágenes crudas

```powershell
python check_dataset.py
```

Reporta cuántas imágenes son legibles, sus resoluciones, y avisa de archivos rotos,
imágenes demasiado pequeñas o en escala de grises. Con `--delete_broken` elimina las rotas.

## 3. Generar los pares A|B

Primero calibra los umbrales de Canny sobre unas pocas imágenes:

```powershell
python preprocess_canny.py --preview 8 --t1 100 --t2 200
```

Mira `./data/preview`: los bordes deben capturar la estructura de las olas sin convertirse
en ruido. En agua suele ayudar suavizar antes del Canny (igual que en el webcam-pix2pix de Akten):

```powershell
python preprocess_canny.py --preview 8 --t1 60 --t2 180 --pre_blur 3 --pre_median 3
```

Cuando el resultado convenza, procesa todo:

```powershell
python preprocess_canny.py --t1 60 --t2 180 --pre_blur 3 --pre_median 3 --val_frac 0.05
```

Salida: `./data/AB/train/*.png`, cada uno de **512x256** = `[A(256) | B(256)]`.

Opciones principales:

| Flag | Qué hace |
|---|---|
| `--t1 / --t2` | umbrales bajo/alto de `cv2.Canny` |
| `--pre_blur N` | `GaussianBlur` de kernel N (impar) antes del Canny |
| `--pre_median N` | `medianBlur` de kernel N (impar) antes del Canny |
| `--aperture` | kernel Sobel de Canny (3/5/7) |
| `--invert` | bordes negros sobre blanco |
| `--size` | lado del recuadro (default 256; el par sale 2·size × size) |
| `--val_frac` | fracción desviada a `data/AB/val` |
| `--preview N` | procesa solo N imágenes a `data/preview` |
| `--overwrite` | reprocesa pares ya existentes |

Las imágenes se **estiran** a 256x256 sin preservar aspect ratio, igual que en el pipeline
original de Akten.

## 4. Entrenamiento (Google Colab, GPU)

Sube `data/AB` y el repo a Colab, y desde `pix2pix/`:

```bash
python train.py --dataroot ./data/AB --name ocean_learning_to_see \
  --model pix2pix --direction AtoB --dataset_mode aligned
```

Flags útiles en Colab:

```bash
  --n_epochs 100 --n_epochs_decay 100   # 200 epochs en total (default)
  --batch_size 4                        # subir si la GPU lo permite
  --display_id -1                       # desactiva visdom (no hay servidor en Colab)
  --save_epoch_freq 20                  # checkpoints periódicos por si se corta la sesión
  --continue_train --epoch_count N      # reanudar tras una desconexión
```

Los checkpoints quedan en `pix2pix/checkpoints/ocean_learning_to_see/`.

## 5. Test / inferencia

```bash
python test.py --dataroot ./data/AB --name ocean_learning_to_see \
  --model pix2pix --direction AtoB --dataset_mode aligned --num_test 50
```

Resultados en `pix2pix/results/ocean_learning_to_see/test_latest/index.html`.

Para inferir sobre bordes arbitrarios (el uso "real" de la pieza), genera pares A|B con
`preprocess_canny.py` a partir de tus nuevas imágenes hacia `data/AB/test` y usa
`--dataroot ./data/AB --phase test`. El lado B se ignora en la generación; solo se usa como
referencia visual en el HTML de resultados.

---

## Notas

- `visdom` no está en `requirements.txt`: su instalación falla en Windows con setuptools
  moderno y el repo ya no lo lista. Usa `--display_id -1`, o `--use_wandb` para logging.
- El repo clonado usa `environment.yml` (conda) en lugar de `requirements.txt`; el
  `requirements.txt` de la raíz es el equivalente pip de esas dependencias más OpenCV.
- `data/` y `checkpoints/` están en `.gitignore`.

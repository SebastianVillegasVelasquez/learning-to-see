# Learning to See

A technical replica of **[Learning to See](https://www.memo.tv/works/learning-to-see/)** by Memo Akten, built with **PyTorch** and **pix2pix**. A generative network that only knows one kind of image (fire, sea, clouds, flowers) tries to reinterpret *anything* you show it through the camera as if it belonged to that world.

![Demo: webcam, Canny edges and generated output](docs/demo_panel.png)

*Left to right: what the webcam sees, the Canny edges the model receives, and the image it "hallucinates". Here, a phone charger as interpreted by the clouds model.*

---

## How it works

The conceptual trick is to train pix2pix on a **single domain** of images. The training pair is not two different images, but the same image and its edge map:

```
A (input)  = cv2.Canny(image)     <- what the model "sees"
B (target) = the original image   <- what the model "imagines"
```

Since the generator has only ever seen fire (or sea, or flowers), when it receives the edges of a rag or a crumpled piece of paper at inference time, it has no choice but to fill that shape with the only material it knows.

```
webcam -> resize 256x256 -> [median -> blur] -> Canny -> A
                                                         |
                                            U-Net 256 generator
                                                         v
                                                hallucination B'
```

| Training pair (A &#124; B) |
|---|
| ![A\|B pair from the fire dataset](docs/sample_pair_fire.png) |

## Trained models

| Model | Key | Images | Canny (t1/t2) | Epochs | Validation L1 |
|---|:---:|---:|:---:|---:|---:|
| Fire | `1` | 682 | 50 / 150 | 100 | **17.3** / 255 |
| Sea | `2` | 522 (filtered) | 20 / 60, blur 5, median 3 | 130 | 36.3 / 255 |
| Clouds | `3` | — | — | — | — |
| Red flowers | `4` | 197 | 40 / 120 | 200 | 26.2 / 255 |

The live Canny thresholds are not the training ones: a webcam scene produces fewer edges than the dataset photos, so each model ships with a hand-calibrated *preset* (see [docs/INFERENCE.md](docs/INFERENCE.md)).

## Features

- **Reproducible data pipeline**: raw dataset validation and A|B pair generation with configurable Canny (`check_dataset.py`, `preprocess_canny.py`).
- **pix2pix training** (cGAN + L1, `unet_256` generator, PatchGAN discriminator) on top of Jun-Yan Zhu's reference repository, included as a submodule.
- **Live webcam inference** with hot model switching (keys `1`-`4`), real-time threshold tuning, screenshots and fullscreen mode.
- **Cross-platform acceleration**: CUDA, Apple Silicon (MPS) or CPU, picked automatically.
- **Acceleration never degrades the image**: on every start, the accelerator output is compared against a CPU reference and discarded if it differs by more than the tolerance. `--half` (fp16) is only enabled if it is correct *and* faster.
- **Performance**: ~62 fps on a GTX 1050 (4 GB) versus ~4 fps on CPU.

## Repository layout

```
learning_to_see/
├── pix2pix/                 # submodule: junyanz/pytorch-CycleGAN-and-pix2pix
├── preprocess_canny.py      # raw images -> A|B pairs
├── check_dataset.py         # validates the dataset before preprocessing
├── webcam_inference.py      # live webcam demo
├── setup_mac.sh             # macOS (Apple Silicon) installer
├── run_mac.sh               # macOS launcher
├── requirements.txt
├── logs/                    # training run logs
└── docs/                    # detailed guides and images
```

The image datasets (`entrenamiento/`, `data/`) and the trained weights (`checkpoints/`, ~200 MB per generator) are **not versioned**; see [Data and models](#data-and-models).

## Installation

```bash
git clone --recurse-submodules https://github.com/SebastianVillegasVelasquez/learning-to-see.git
cd learning-to-see
```

If you already cloned without submodules: `git submodule update --init`.

**Windows / Linux**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate

# NVIDIA GPU (CUDA 12.6)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

**macOS (Apple Silicon)**

```bash
bash setup_mac.sh
```

The standard PyTorch build for macOS already includes MPS; no special index is needed.

## Usage

### 1. Prepare the dataset

```bash
python check_dataset.py --input_dir ./data/raw_images
python preprocess_canny.py --preview 8 --t1 60 --t2 180 --pre_blur 3 --pre_median 3   # calibrate
python preprocess_canny.py --t1 60 --t2 180 --pre_blur 3 --pre_median 3 --val_frac 0.05
```

This produces `data/AB/train/*.png`, each one 512x256 = `[A | B]`.

### 2. Train

From `pix2pix/`:

```bash
python train.py --dataroot ../data/AB --name my_model \
  --model pix2pix --direction AtoB --dataset_mode aligned \
  --load_size 256 --crop_size 256 --batch_size 1 \
  --n_epochs 65 --n_epochs_decay 65 --save_epoch_freq 10 \
  --checkpoints_dir ../checkpoints
```

Details and recommendations in [docs/TRAINING.md](docs/TRAINING.md).

### 3. Live inference

```bash
python webcam_inference.py                    # Windows / Linux
bash run_mac.sh                               # macOS
python webcam_inference.py --benchmark 50     # fps without needing a webcam
```

| Key | Action |
|---|---|
| `1` `2` `3` `4` | switch model (with its filtering preset) |
| `+` `-` | raise / lower both Canny thresholds |
| `b` / `m` | cycle the pre-Canny Gaussian blur / median filter |
| `i` · `f` · `r` | invert edges · mirror · restore preset |
| `s` | save a screenshot |
| `h` · `q` | toggle help · quit |

The full option list, camera troubleshooting and performance measurements are in [docs/INFERENCE.md](docs/INFERENCE.md).

## Data and models

- **Datasets**: custom image sets per domain (fire, sea, flowers). They are not included because of their size (~550 MB, more than 9,500 images). To reproduce a model, put your images in `data/raw_images/` and follow step 1.
- **Weights**: `checkpoints/<model>/latest_net_G.pth`. The inference script looks for them in `checkpoints/fire`, `sea`, `v2` (clouds) and `flores_rojas`.

## Design decisions

- **One domain, not two.** This is what gives the piece its meaning: the model does not translate between styles, it "fills in" edges with the only thing it knows.
- **Stretch to 256x256 without preserving aspect ratio**, as in the original pipeline, so preprocessing and inference match exactly.
- **Normalization identical to training** (`(x - 0.5) / 0.5`), and the generator is built with the same factory from `pix2pix/models/networks.py`.
- **Deliberately simple camera capture.** Attempts to speed it up with threads or forced backends made the image worse on Windows; the reasoning is documented in [docs/INFERENCE.md](docs/INFERENCE.md).

## Credits

- **Memo Akten**, author of [Learning to See](https://www.memo.tv/works/learning-to-see/) and [webcam-pix2pix-tensorflow](https://github.com/memoakten/webcam-pix2pix-tensorflow), where the preprocessing pattern comes from.
- **Jun-Yan Zhu et al.**, [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
- Isola et al., [*Image-to-Image Translation with Conditional Adversarial Networks*](https://arxiv.org/abs/1611.07004), CVPR 2017.

## License

This project is released under the [MIT](LICENSE) license. The `pix2pix/` submodule keeps its own license (BSD) and the image datasets belong to their respective authors.

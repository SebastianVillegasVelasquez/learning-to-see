# Live inference — `webcam_inference.py`

A functional replica of Memo Akten's [webcam-pix2pix-tensorflow](https://github.com/memoakten/webcam-pix2pix-tensorflow),
with pix2pix generators trained in PyTorch.

```
webcam ──> resize 256×256 ──> [median → blur] ──> Canny ──> A
                                                            │
                                              G (unet_256)  ▼
                                                    hallucination B'
```

Each model knows a single domain, so it translates **any** edge input into that domain.
pix2pix does not replace the shape you give it: it fills it with the material it learned.

---

## Quick start

### macOS (Apple Silicon, M1–M5)

```bash
bash setup_mac.sh     # creates the environment and installs everything (once)
bash run_mac.sh       # menu to pick a model and start
```

`setup_mac.sh` checks the architecture, creates `.venv`, installs PyTorch (the standard macOS
build **already ships MPS**, no special index needed), clones `pix2pix` if missing and
checks that the GPU responds with a real operation.

The first time, macOS will ask for camera permission for your terminal. If the dialog does not appear:
**System Settings › Privacy & Security › Camera**.

### Windows

```powershell
.\.venv\Scripts\Activate.ps1
python webcam_inference.py
```

### Manual, with any model

```bash
python webcam_inference.py --checkpoint ./checkpoints/fire/latest_net_G.pth
```

---

## Models: switched live with keys 1-4

Each key loads its model **and its calibrated filtering** at once. There is no need to remember
thresholds or restart anything.

| Key | Model | t1 / t2 | blur / median | Try it with |
|---|---|---|---|---|
| `1` | **Fire** | 50 / 75 | off / off | crumpled paper |
| `2` | **Sea** | 20 / 40 | 3 / off | a rag, a shirt |
| `3` | **Clouds** | 50 / 70 | off / off | crumpled paper |
| `4` | **Flowers** | 25 / 30 | 5 / 3 | crumpled paper simulating a petal |

Models are loaded the first time their key is pressed and stay in memory, so from then on
switching is instant. All four together take **847 MB** of VRAM.

These thresholds are not the training ones but the ones calibrated live: a webcam
scene produces fewer edges than the dataset photos, and lowering them compensates for that difference.

---

## Keyboard controls

| Key | Action |
|---|---|
| `1` `2` `3` `4` | switch model (with its filtering) |
| `h` | text: minimal → full help → none |
| `+` / `-` | raise/lower **both** Canny thresholds |
| `[` / `]` | lower/raise `t1` only |
| `,` / `.` | lower/raise `t2` only |
| `b` | cycle the pre-Canny `GaussianBlur`: 0 → 3 → 5 → 7 |
| `m` | cycle the pre-Canny `medianBlur`: 0 → 3 → 5 → 7 |
| `i` | invert the edges |
| `f` | toggle mirror |
| `r` | restore the current model's preset |
| `s` | save a clean capture to `./screenshots/` |
| `q` / `ESC` | quit |

The `s` capture saves only the generated image, with no text or screen scaling.

## Presentation

It starts **fullscreen** showing only the hallucination. The image is square
and is centered with black bars (`WINDOW_KEEPRATIO`), without distortion on 16:9 screens.

- `--ventana` to open in a normal window.
- `--hud 0` to start with no text at all; `1` (default) shows the model name
  and parameters in small type; `2` shows the key list.
- `--display_size N` changes the resolution it is scaled to before display (900 by
  default). It only affects what is shown, never the inference, which is always at 256×256.

If the image comes out empty or almost black, the camera is pointing at a smooth surface: the
model has no edges to interpret. Point it at something with texture.

---

## Hardware acceleration

`--device auto` (default) tries, in this order:

| Order | Backend | Hardware |
|---|---|---|
| 1 | **CUDA** | NVIDIA GPU |
| 2 | **MPS** | Apple Silicon GPU (Metal) |
| 3 | **CPU** | anything |

Force one with `--device cuda|mps|cpu`. If you ask for one that is not available, it fails with a message
explaining what is missing instead of silently falling back to CPU.

### Acceleration never degrades the image

It is implemented as a real check on every start, not as a promise:

1. Reference forward pass **on CPU**, in float32.
2. The same forward pass **on the accelerator**.
3. Compares both outputs and reports the maximum difference, also in color levels out of 255.

In `eval()` the generator is deterministic (dropout off, BatchNorm with fixed statistics),
so both outputs must match up to rounding. If the difference exceeds `--verify_tol`
(0.02 ≈ 2.5 color levels), **the accelerator is discarded automatically** and it continues on
CPU. The same happens if the backend raises an error, a realistic case on MPS when an operation is not
implemented in Metal.

Measured on a GTX 1050: difference of `3.52e-06`, i.e. **0.00 color levels**.

### `--half`: measured, not assumed

```bash
python webcam_inference.py --half
```

float16 passes **two** filters before being used:

1. **Quality** — compared against the CPU float32 reference. If it exceeds the tolerance,
   it reverts to float32 with a warning.
2. **Speed** — both precisions are timed and the faster one is kept.

The second filter exists because fp16 **does not always speed things up**. Measured on a GTX 1050
(Pascal): fp32 14.7 ms versus fp16 16.1 ms, so fp16 is **8% slower**, even though
it is numerically correct (0.44 color levels of difference). Pascal has no fast fp16
units. On Apple Silicon the opposite is usually true.

Actual output on the 1050 (the program prints it in Spanish):

```
  Verificación fp16  : dif. máx. vs CPU = 3.49e-03 (0.44 niveles de color de 255)
  Velocidad          : fp32 14.7 ms | fp16 16.1 ms
  [i] fp16 no acelera en este hardware: se mantiene float32.
```

That is why `run_mac.sh` can enable it safely: if it did not help on the M5, the script
itself discards it.

### Camera capture: deliberately simple

```python
cap = cv2.VideoCapture(args.camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
...
ok, frame = cap.read()
```

That is all, and it is intentional. **Do not touch this part to gain fps.** Three
variants were tried and all three made things worse on Windows:

| Attempt | Result |
|---|---|
| Capture in a separate thread | Banded RGB noise: the backend reuses the buffer between reads |
| Force backend (MSMF/DSHOW) + `CAP_PROP_BUFFERSIZE` | The camera stopped opening |
| `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` + retries | RGB noise, 1 fps and erratic startup |

OpenCV already picks the right backend on each system. If the camera fails, the problem is
outside the script: almost always another application is holding it (Teams, Zoom, the Windows
Camera app, a browser tab), even in the background.

### Applied optimizations (none of them affect the image)

- `torch.inference_mode()` instead of `no_grad()` — slightly faster, identical result.
- `cudnn.benchmark` on CUDA — caches the fastest convolution algorithm for the
  fixed 256×256 resolution.
- Warm-up of 3 passes at startup, so the first fps readings are not skewed by
  kernel compilation.
- Automatic precision selection (see `--half`), verified for quality and speed.

Quantization and `channels_last` are **not** used, because they could change the image.

---

## Diagnostics

```bash
python webcam_inference.py --benchmark 50     # fps without needing a webcam
```

### If the camera does not start on Windows

Typical error:

```
videoio(MSMF): can't grab frame. Error: -1072875772
[!] Se perdió el stream de la cámara.
```

`-1072875772` is `0xC00D3704` = `MF_E_HW_MFT_FAILED_START_STREAMING`: the camera opens but
Media Foundation cannot start the stream. **It is a system problem, not a script
problem.** In order of likelihood:

1. **Another application has the camera.** By far the most frequent cause. Close Teams,
   Zoom, the Windows Camera app and browser tabs with camera permission; it is enough
   for them to be in the background. Check it in Settings › Privacy › Camera, under
   "Recent activity".
2. **Hung driver state.** Restart the computer. It is what resolves it most often.
3. **Another camera index.** Try `--camera 1`.

Reference measurements (`unet_256`, 256×256, float32):

| Backend | Mean | fps |
|---|---|---|
| CUDA · GTX 1050 4 GB | 16.0 ms | 62.5 |
| CPU · 4 threads | 250 ms | 4.0 |

---

## Options

| Flag | Default | What it is for |
|---|---|---|
| `--checkpoint` | autodetect | path to the generator |
| `--camera N` | `0` | another webcam |
| `--device` | `auto` | `auto` / `cuda` / `mps` / `cpu` |
| `--half` | — | verified float16 |
| `--benchmark N` | `0` | measure N passes and exit |
| `--ventana` | — | normal window instead of fullscreen |
| `--hud {0,1,2}` | `1` | text shown at startup |
| `--display_size N` | `900` | display resolution |
| `--no_mirror` | — | no mirror effect |
| `--verify_tol` | `0.02` | tolerance against CPU |
| `--no_verify` | — | skip the verification |
| `--step N` | `5` | sensitivity of `+`/`-` |

---

## Implementation notes

- **No data pipeline.** Only `pix2pix/models/networks.py` is imported, which depends
  solely on `torch`. `options/` and `data/` are not touched, so neither
  `--dataroot` nor `--dataset_mode` is needed.
- **DataParallel's `module.` prefix.** `strip_module_prefix()` removes the prefix if it appears.
- **Tolerant loading.** `load_state_dict(strict=False)` reporting missing or
  unexpected keys: if there are many, the architecture does not match the checkpoint.
- **Normalization identical to training:** RGB → `[0,1]` → `(x-0.5)/0.5`, the same as
  `ToTensor()+Normalize((0.5,)*3,(0.5,)*3)` in `pix2pix/data/base_dataset.py`. The inverse
  replicates `util.tensor2im`.
- The resize is a direct stretch to 256×256 without preserving the aspect ratio, as in
  `preprocess_canny.py` and in Akten's original pipeline.

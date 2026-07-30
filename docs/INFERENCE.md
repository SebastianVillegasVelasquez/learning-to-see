# Inferencia en vivo — `webcam_inference.py`

Réplica funcional de [webcam-pix2pix-tensorflow](https://github.com/memoakten/webcam-pix2pix-tensorflow)
de Memo Akten, con generadores pix2pix entrenados en PyTorch.

```
webcam ──> resize 256×256 ──> [median → blur] ──> Canny ──> A
                                                            │
                                              G (unet_256)  ▼
                                                       alucinación B'
```

Cada modelo conoce un solo dominio, así que traduce **cualquier** entrada de bordes a ese
dominio. pix2pix no sustituye la forma que le das: la rellena con la materia que aprendió.

---

## Arranque rápido

### macOS (Apple Silicon, M1–M5)

```bash
bash setup_mac.sh     # crea el entorno e instala todo (una sola vez)
bash run_mac.sh       # menú para elegir modelo y arrancar
```

`setup_mac.sh` verifica la arquitectura, crea `.venv`, instala PyTorch (la build estándar de
macOS **ya trae MPS**, no hace falta ningún índice especial), clona `pix2pix` si falta y
comprueba que la GPU responde con una operación real.

La primera vez macOS pedirá permiso de cámara para tu terminal. Si no aparece el diálogo:
**Ajustes del Sistema › Privacidad y seguridad › Cámara**.

### Windows

```powershell
.\.venv\Scripts\Activate.ps1
python webcam_inference.py
```

### Manual, con cualquier modelo

```bash
python webcam_inference.py --checkpoint ./checkpoints/fire/latest_net_G.pth
```

---

## Modelos: se cambian en vivo con las teclas 1-4

Cada tecla carga su modelo **y su filtrado calibrado** a la vez. No hay que recordar
umbrales ni reiniciar nada.

| Tecla | Modelo | t1 / t2 | blur / median | Probar con |
|---|---|---|---|---|
| `1` | **Fuego** | 50 / 75 | off / off | papel arrugado |
| `2` | **Mar** | 20 / 40 | 3 / off | un trapo, una camisa |
| `3` | **Nubes** | 50 / 70 | off / off | papel arrugado |
| `4` | **Flores** | 25 / 30 | 5 / 3 | papel arrugado simulando un pétalo |

Los modelos se cargan la primera vez que se pulsa su tecla y quedan en memoria, así que a
partir de ahí el cambio es instantáneo. Los cuatro juntos ocupan **847 MB** de VRAM.

Estos umbrales no son los del entrenamiento sino los calibrados en vivo: una escena de
webcam produce menos bordes que las fotos del dataset, y bajarlos compensa esa diferencia.

---

## Controles de teclado

| Tecla | Acción |
|---|---|
| `1` `2` `3` `4` | cambiar de modelo (con su filtrado) |
| `h` | textos: mínimo → ayuda completa → sin nada |
| `+` / `-` | subir/bajar **ambos** umbrales de Canny |
| `[` / `]` | bajar/subir solo `t1` |
| `,` / `.` | bajar/subir solo `t2` |
| `b` | ciclar `GaussianBlur` previo: 0 → 3 → 5 → 7 |
| `m` | ciclar `medianBlur` previo: 0 → 3 → 5 → 7 |
| `i` | invertir los bordes |
| `f` | activar/desactivar espejo |
| `r` | restaurar el preset del modelo actual |
| `s` | guardar captura limpia en `./screenshots/` |
| `q` / `ESC` | salir |

La captura con `s` guarda solo la imagen generada, sin textos ni escalado de pantalla.

## Presentación

Arranca **a pantalla completa** mostrando únicamente la alucinación. La imagen es cuadrada
y se centra con bandas negras (`WINDOW_KEEPRATIO`), sin deformarse en pantallas 16:9.

- `--ventana` para abrir en ventana normal.
- `--hud 0` para arrancar sin ningún texto; `1` (por defecto) muestra el nombre del modelo
  y los parámetros en letra pequeña; `2` muestra la lista de teclas.
- `--display_size N` cambia la resolución a la que se escala antes de mostrar (900 por
  defecto). Solo afecta a lo que se ve, nunca a la inferencia, que siempre es a 256×256.

Si la imagen sale vacía o casi negra, la cámara está enfocando una superficie lisa: el
modelo no tiene bordes que interpretar. Apunta a algo con textura.

---

## Aceleración por hardware

`--device auto` (por defecto) prueba en este orden:

| Orden | Backend | Hardware |
|---|---|---|
| 1 | **CUDA** | GPU NVIDIA |
| 2 | **MPS** | GPU Apple Silicon (Metal) |
| 3 | **CPU** | cualquier cosa |

Fuerza uno con `--device cuda|mps|cpu`. Si pides uno no disponible, falla con un mensaje que
explica qué falta en lugar de caer a CPU en silencio.

### La aceleración nunca degrada la imagen

Está implementado como una comprobación real en cada arranque, no como una promesa:

1. Forward pass de referencia **en CPU**, en float32.
2. El mismo forward pass **en el acelerador**.
3. Compara ambas salidas y reporta la diferencia máxima, también en niveles de color de 255.

En `eval()` el generador es determinista (dropout apagado, BatchNorm con estadísticas fijas),
así que ambas salidas deben coincidir salvo redondeo. Si la diferencia supera `--verify_tol`
(0,02 ≈ 2,5 niveles de color), **el acelerador se descarta automáticamente** y se continúa en
CPU. Igual si el backend lanza un error, caso realista en MPS cuando una operación no está
implementada en Metal.

Medido en una GTX 1050: diferencia de `3,52e-06`, es decir **0,00 niveles de color**.

### `--half`: se mide, no se supone

```bash
python webcam_inference.py --half
```

float16 pasa **dos** filtros antes de usarse:

1. **Calidad** — se compara contra la referencia en CPU float32. Si supera la tolerancia,
   se revierte a float32 avisando.
2. **Velocidad** — se cronometran ambas precisiones y se queda con la más rápida.

El segundo filtro existe porque fp16 **no siempre acelera**. Medido en una GTX 1050
(Pascal): fp32 14,7 ms frente a fp16 16,1 ms, o sea fp16 es un **8 % más lento**, aunque
numéricamente sea correcto (0,44 niveles de color de diferencia). Pascal no tiene unidades
fp16 rápidas. En Apple Silicon lo normal es lo contrario.

Salida real en la 1050:

```
  Verificación fp16  : dif. máx. vs CPU = 3.49e-03 (0.44 niveles de color de 255)
  Velocidad          : fp32 14.7 ms | fp16 16.1 ms
  [i] fp16 no acelera en este hardware: se mantiene float32.
```

Por eso `run_mac.sh` puede activarlo sin riesgo: si en el M5 no ayudase, el propio script
lo descarta.

### Captura de cámara: deliberadamente simple

```python
cap = cv2.VideoCapture(args.camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
...
ok, frame = cap.read()
```

Eso es todo, y es intencionado. **No toques esta parte para ganar fps.** Se probaron tres
variantes y las tres empeoraron las cosas en Windows:

| Intento | Resultado |
|---|---|
| Captura en un hilo aparte | Ruido RGB con bandas: el backend reutiliza el búfer entre lecturas |
| Forzar backend (MSMF/DSHOW) + `CAP_PROP_BUFFERSIZE` | La cámara dejaba de abrir |
| `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` + reintentos | Ruido RGB, 1 fps y arranque errático |

OpenCV ya elige el backend adecuado en cada sistema. Si la cámara falla, el problema está
fuera del script: casi siempre otra aplicación la tiene tomada (Teams, Zoom, la app Cámara
de Windows, una pestaña del navegador), incluso en segundo plano.

### Optimizaciones aplicadas (todas sin efecto en la imagen)

- `torch.inference_mode()` en lugar de `no_grad()` — algo más rápido, resultado idéntico.
- `cudnn.benchmark` en CUDA — cachea el algoritmo de convolución más rápido para la
  resolución fija de 256×256.
- Warm-up de 3 pasadas al arrancar, para que los primeros fps no salgan falseados por la
  compilación de kernels.
- Selección automática de precisión (ver `--half`), verificada en calidad y en velocidad.

**No** se usa cuantización ni `channels_last`, porque podrían cambiar la imagen.

---

## Diagnóstico

```bash
python webcam_inference.py --benchmark 50     # fps sin necesitar webcam
```

### Si la cámara no arranca en Windows

Error típico:

```
videoio(MSMF): can't grab frame. Error: -1072875772
[!] Se perdió el stream de la cámara.
```

`-1072875772` es `0xC00D3704` = `MF_E_HW_MFT_FAILED_START_STREAMING`: la cámara se abre pero
Media Foundation no consigue iniciar el stream. **Es un problema del sistema, no del
script.** Por orden de probabilidad:

1. **Otra aplicación tiene la cámara.** La causa más frecuente con diferencia. Cierra Teams,
   Zoom, la app Cámara de Windows y las pestañas del navegador con permiso de cámara; basta
   con que estén en segundo plano. Compruébalo en Configuración › Privacidad › Cámara, en
   «Actividad reciente».
2. **Estado colgado del driver.** Reinicia el equipo. Es lo que más veces lo resuelve.
3. **Otro índice de cámara.** Prueba `--camera 1`.

Medidas de referencia (`unet_256`, 256×256, float32):

| Backend | Media | fps |
|---|---|---|
| CUDA · GTX 1050 4 GB | 16,0 ms | 62,5 |
| CPU · 4 hilos | 250 ms | 4,0 |

---

## Opciones

| Flag | Por defecto | Para qué |
|---|---|---|
| `--checkpoint` | autodetecta | ruta al generador |
| `--camera N` | `0` | otra webcam |
| `--device` | `auto` | `auto` / `cuda` / `mps` / `cpu` |
| `--half` | — | float16 verificado |
| `--benchmark N` | `0` | medir N pasadas y salir |
| `--ventana` | — | ventana normal en vez de pantalla completa |
| `--hud {0,1,2}` | `1` | textos al arrancar |
| `--display_size N` | `900` | resolución de visualización |
| `--no_mirror` | — | sin efecto espejo |
| `--verify_tol` | `0.02` | tolerancia frente a CPU |
| `--no_verify` | — | omitir la verificación |
| `--step N` | `5` | sensibilidad de `+`/`-` |

---

## Notas de implementación

- **Sin pipeline de datos.** Solo se importa `pix2pix/models/networks.py`, que depende
  únicamente de `torch`. No se tocan `options/` ni `data/`, así que no hacen falta
  `--dataroot` ni `--dataset_mode`.
- **`module.` de DataParallel.** `strip_module_prefix()` limpia el prefijo si aparece.
- **Carga tolerante.** `load_state_dict(strict=False)` reportando claves faltantes o
  sobrantes: si son muchas, la arquitectura no coincide con el checkpoint.
- **Normalización idéntica al entrenamiento:** RGB → `[0,1]` → `(x-0.5)/0.5`, igual que
  `ToTensor()+Normalize((0.5,)*3,(0.5,)*3)` en `pix2pix/data/base_dataset.py`. La inversa
  replica `util.tensor2im`.
- El resize es un estirado directo a 256×256 sin preservar aspect ratio, igual que en
  `preprocess_canny.py` y que en el pipeline original de Akten.

"""
webcam_inference.py — inferencia en vivo estilo "Learning to See" (Memo Akten).

Réplica funcional de memoakten/webcam-pix2pix-tensorflow, pero cargando el generador
entrenado con pytorch-CycleGAN-and-pix2pix.

Pipeline por frame:
    webcam -> resize 256x256 -> [median/blur] -> Canny  =  A (lo que el modelo "ve")
    A -> tensor [1,3,256,256] en [-1,1] -> G -> B'      =  la alucinación

El preprocesado replica exactamente el de `preprocess_canny.py` (usado para construir el
dataset de entrenamiento), y la normalización replica la de `data/base_dataset.py` del
repo pix2pix: ToTensor + Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)).

IMPORTANTE: los umbrales de Canny en vivo deberían coincidir con los que usaste al
generar el dataset. Si no coinciden, el modelo recibe bordes fuera de la distribución que
aprendió y alucina peor. Usa +/- para calibrarlos en caliente.

Uso:
    python webcam_inference.py
    python webcam_inference.py --t1 60 --t2 180 --pre_blur 3 --pre_median 3
    python webcam_inference.py --checkpoint ./checkpoints/latest_net_G.pth --camera 1
"""

import argparse
import platform
import sys
import time
from collections import OrderedDict, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

IS_MAC = platform.system() == "Darwin"

PROJECT_ROOT = Path(__file__).resolve().parent

# El generador se construye con la MISMA fábrica que usó el entrenamiento, importando solo
# models/networks.py. Ese módulo depende únicamente de torch (no de options/ ni de data/),
# así que evitamos por completo dataroot, dataset_mode y el resto del pipeline de datos.
# Se admiten dos disposiciones: el repo pix2pix clonado al lado (desarrollo) o una copia
# de networks.py en ./models (paquete autocontenido, sin necesidad de git).
for _candidato in (PROJECT_ROOT / "pix2pix", PROJECT_ROOT):
    if (_candidato / "models" / "networks.py").is_file():
        sys.path.insert(0, str(_candidato))
        break
try:
    from models import networks  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"No se pudo importar models/networks.py ({exc}).\n"
        f"Se buscó en:\n"
        f"  {PROJECT_ROOT / 'pix2pix' / 'models' / 'networks.py'}\n"
        f"  {PROJECT_ROOT / 'models' / 'networks.py'}\n"
        "Si tienes git: git clone --depth 1 "
        "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git pix2pix"
    )

# Carpeta de modelos, siempre relativa al script: así funciona aunque se lance desde otro
# directorio (doble clic, acceso directo, terminal abierta en otro sitio).
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Rutas donde buscar el checkpoint si no se pasa --checkpoint explícito, en orden de
# preferencia. Cada modelo exige sus propios umbrales Canny (ver MODELOS).
DEFAULT_CKPT_CANDIDATES = [
    CHECKPOINTS_DIR / "fire" / "latest_net_G.pth",
    CHECKPOINTS_DIR / "sea" / "latest_net_G.pth",
    CHECKPOINTS_DIR / "v2" / "latest_net_G.pth",
    CHECKPOINTS_DIR / "flores_rojas" / "latest_net_G.pth",
    CHECKPOINTS_DIR / "latest_net_G.pth",
]

# Modelos disponibles, en el orden de las teclas 1..4. Los umbrales NO son los del
# entrenamiento sino los calibrados en vivo con webcam: una escena real da menos bordes
# que las fotos del dataset, así que bajarlos compensa esa diferencia.
MODELOS = [
    dict(tecla="1", nombre="Fuego", carpeta="fire",
         t1=50, t2=75, pre_blur=0, pre_median=0,
         pista="papel arrugado"),
    dict(tecla="2", nombre="Mar", carpeta="sea",
         t1=20, t2=40, pre_blur=3, pre_median=0,
         pista="trapo o camisa"),
    dict(tecla="3", nombre="Nubes", carpeta="v2",
         t1=50, t2=70, pre_blur=0, pre_median=0,
         pista="papel arrugado"),
    dict(tecla="4", nombre="Flores", carpeta="flores_rojas",
         t1=25, t2=30, pre_blur=5, pre_median=3,
         pista="papel arrugado simulando un petalo"),
]
PRESETS = {m["carpeta"]: m for m in MODELOS}
TECLAS = {ord(m["tecla"]): m for m in MODELOS}

BLUR_CYCLE = [0, 3, 5, 7]  # valores que recorren las teclas 'b' y 'm'


def parse_args():
    p = argparse.ArgumentParser(
        description="Inferencia pix2pix en vivo desde webcam (Learning to See).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- modelo ---
    p.add_argument("--checkpoint", default=None,
                   help="Ruta a latest_net_G.pth. Si se omite, se busca en rutas conocidas.")
    p.add_argument("--netG", default="unet_256",
                   choices=["unet_256", "unet_128", "resnet_9blocks", "resnet_6blocks"],
                   help="Arquitectura del generador (debe coincidir con el entrenamiento).")
    p.add_argument("--ngf", type=int, default=64, help="Filtros base del generador.")
    p.add_argument("--norm", default="batch", choices=["batch", "instance", "none"],
                   help="Normalización usada en entrenamiento.")
    p.add_argument("--input_nc", type=int, default=3)
    p.add_argument("--output_nc", type=int, default=3)
    p.add_argument("--no_dropout", action="store_true",
                   help="Marca esto solo si entrenaste con --no_dropout.")

    # --- dispositivo / aceleración ---
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                   help="Acelerador. 'auto' prueba CUDA (NVIDIA) -> MPS (Apple Silicon) -> CPU.")
    p.add_argument("--cpu", action="store_true",
                   help="Atajo equivalente a --device cpu.")
    p.add_argument("--no_verify", action="store_true",
                   help="Omite la comprobación numérica acelerador-vs-CPU del arranque.")
    p.add_argument("--verify_tol", type=float, default=2e-2,
                   help="Diferencia máx. tolerada (escala [-1,1]) frente a la salida en CPU. "
                        "Por encima, se descarta el acelerador y se usa CPU.")
    p.add_argument("--benchmark", type=int, default=0, metavar="N",
                   help="Si >0, mide N forward passes y sale. No necesita webcam.")
    p.add_argument("--half", action="store_true",
                   help="Usa float16 en el acelerador. Bastante más rápido en Apple Silicon "
                        "y en GPUs con tensor cores. Solo se activa si pasa la verificación "
                        "numérica contra la CPU; si no, se revierte a float32 solo.")

    # --- cámara ---
    p.add_argument("--camera", type=int, default=0, help="Índice de cv2.VideoCapture.")
    p.add_argument("--cam_width", type=int, default=640, help="Ancho de captura solicitado.")
    p.add_argument("--cam_height", type=int, default=480, help="Alto de captura solicitado.")
    p.add_argument("--no_mirror", action="store_true",
                   help="Desactiva el volteo horizontal tipo espejo.")

    # --- preprocesado Canny (debe replicar preprocess_canny.py) ---
    p.add_argument("--size", type=int, default=256, help="Resolución de entrada del modelo.")
    p.add_argument("--t1", type=int, default=100, help="Umbral bajo de Canny.")
    p.add_argument("--t2", type=int, default=200, help="Umbral alto de Canny.")
    p.add_argument("--aperture", type=int, default=3, choices=[3, 5, 7],
                   help="Kernel Sobel de Canny.")
    p.add_argument("--l2_gradient", action="store_true", help="Norma L2 en Canny.")
    p.add_argument("--pre_blur", type=int, default=0,
                   help="GaussianBlur (kernel impar) antes del Canny. 0 = off.")
    p.add_argument("--pre_median", type=int, default=0,
                   help="medianBlur (kernel impar) antes del Canny. 0 = off.")
    p.add_argument("--invert", action="store_true", help="Bordes negros sobre blanco.")

    # --- salida ---
    p.add_argument("--ventana", action="store_true",
                   help="Abre en ventana normal en vez de pantalla completa.")
    p.add_argument("--hud", type=int, default=1, choices=[0, 1, 2],
                   help="Textos en pantalla: 0 nada, 1 mínimo, 2 ayuda completa. "
                        "Se cambia en vivo con la tecla 'h'.")
    p.add_argument("--display_size", type=int, default=900,
                   help="Lado en píxeles al que se escala la imagen antes de mostrarla.")
    p.add_argument("--screenshot_dir", default="./screenshots",
                   help="Carpeta donde guardar capturas con la tecla 's'.")
    p.add_argument("--step", type=int, default=5,
                   help="Incremento de los umbrales por pulsación de tecla.")
    return p.parse_args()


# --------------------------------------------------------------------------------------
# Cámara
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# Selección de dispositivo (CUDA / Apple Silicon MPS / CPU)
# --------------------------------------------------------------------------------------

def mps_available() -> bool:
    """MPS = Metal Performance Shaders, el backend GPU de PyTorch en Apple Silicon.
    Se consulta con hasattr porque torch < 1.12 no tiene torch.backends.mps."""
    backend = getattr(torch.backends, "mps", None)
    return bool(backend and backend.is_available() and backend.is_built())


def pick_device(args) -> torch.device:
    """Resuelve --device. En 'auto': CUDA -> MPS -> CPU."""
    requested = "cpu" if args.cpu else args.device

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "Pediste --device cuda pero torch no ve ninguna GPU NVIDIA.\n"
                f"  torch {torch.__version__} | compilado con CUDA: {torch.version.cuda}\n"
                "Si tu torch es la build CPU (sufijo '+cpu'), reinstálalo con soporte CUDA "
                "desde https://pytorch.org/get-started/locally/")
        return torch.device("cuda")

    if requested == "mps":
        if not mps_available():
            raise SystemExit("Pediste --device mps pero no está disponible. "
                             "MPS requiere macOS con Apple Silicon y torch >= 1.12.")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        return (f"CUDA · {props.name} · {props.total_memory / 1024**3:.1f} GB · "
                f"compute {props.major}.{props.minor}")
    if device.type == "mps":
        return "MPS · GPU Apple Silicon (Metal)"
    return f"CPU · {torch.get_num_threads()} hilos"


def sync(device: torch.device):
    """Los kernels de GPU son asíncronos: hay que sincronizar para medir tiempos reales."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def tune_backend(device: torch.device):
    """Ajustes seguros que no alteran el resultado, solo la velocidad."""
    if device.type == "cuda":
        # La resolución de entrada es fija (256x256), así que cuDNN puede cachear el
        # algoritmo más rápido para esta forma en vez de re-elegirlo en cada frame.
        torch.backends.cudnn.benchmark = True


def time_forward(net: torch.nn.Module, probe: torch.Tensor, device: torch.device,
                 dtype: torch.dtype, n: int = 12) -> float:
    """Tiempo medio de un forward pass, ya calentado. Devuelve segundos."""
    x = probe.to(device=device, dtype=dtype)
    with torch.inference_mode():
        for _ in range(4):        # warm-up: la 1a pasada compila kernels
            net(x)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(n):
            net(x)
        sync(device)
    return (time.perf_counter() - t0) / n


def verify_device(net: torch.nn.Module, device: torch.device, args) -> tuple[torch.device, torch.dtype]:
    """Comprueba que el acelerador da el MISMO resultado que la CPU, y hace warm-up.

    En eval() el generador es determinista (dropout apagado, BatchNorm con estadísticas
    fijas), así que la salida en GPU debe coincidir con la de CPU salvo error de redondeo
    en float32 (~1e-5). Si la diferencia se dispara, el acelerador está degradando la
    imagen y lo descartamos: preferimos 2 fps correctos a 30 fps con artefactos.
    """
    if device.type == "cpu":
        if args.half:
            print("  [i] --half se ignora en CPU (float16 en CPU no acelera).")
        return device, torch.float32

    # Entrada determinista y con estructura (no ruido) para que la comparación sea sensible.
    gen = torch.Generator().manual_seed(0)
    probe = torch.rand(1, args.input_nc, args.size, args.size, generator=gen) * 2 - 1

    with torch.inference_mode():
        ref = net(probe.to("cpu")).cpu().float()

        net.to(device)
        try:
            out = net(probe.to(device))
            sync(device)
            out = out.cpu().float()
        except (RuntimeError, NotImplementedError) as exc:
            # Típico en MPS si alguna op del modelo no está implementada en Metal.
            print(f"  [!] {device.type.upper()} falló al ejecutar el generador: {exc}")
            print("      -> se descarta el acelerador y se continúa en CPU.")
            net.to("cpu")
            return torch.device("cpu"), torch.float32

    max_diff = (out - ref).abs().max().item()
    # Referencia: la salida vive en [-1,1] y se cuantiza a 256 niveles, así que un paso
    # de color equivale a ~0.0078. Por debajo de eso la diferencia es invisible.
    print(f"  Verificación fp32  : dif. máx. vs CPU = {max_diff:.2e} "
          f"({max_diff / (2 / 255):.2f} niveles de color de 255)")

    if max_diff > args.verify_tol:
        print(f"  [!] Supera la tolerancia ({args.verify_tol:.0e}): {device.type.upper()} "
              f"altera la imagen. Descartado, se usa CPU.")
        net.to("cpu")
        return torch.device("cpu"), torch.float32

    print(f"  [ok] {device.type.upper()} numéricamente equivalente a CPU.")

    dtype = torch.float32
    if args.half:
        # Mismo criterio para fp16: solo se acepta si la imagen no cambia de forma
        # perceptible. La referencia sigue siendo la CPU en float32.
        try:
            net.half()
            with torch.inference_mode():
                out16 = net(probe.to(device=device, dtype=torch.float16))
                sync(device)
                out16 = out16.cpu().float()
            diff16 = (out16 - ref).abs().max().item()
            print(f"  Verificación fp16  : dif. máx. vs CPU = {diff16:.2e} "
                  f"({diff16 / (2 / 255):.2f} niveles de color de 255)")
            if diff16 > args.verify_tol:
                print(f"  [!] fp16 supera la tolerancia: se revierte a float32.")
                net.float()
            else:
                # fp16 pasa la prueba de calidad, pero no siempre es más rápido: en GPUs
                # Pascal (GTX 10xx) es más lento que fp32. En vez de suponerlo, se mide
                # y se elige el ganador. Así el mismo código acierta en cualquier equipo.
                t16 = time_forward(net, probe, device, torch.float16)
                net.float()
                t32 = time_forward(net, probe, device, torch.float32)
                print(f"  Velocidad          : fp32 {t32*1000:.1f} ms | "
                      f"fp16 {t16*1000:.1f} ms")
                if t16 < t32 * 0.97:      # margen para no cambiar por ruido de medida
                    net.half()
                    dtype = torch.float16
                    print(f"  [ok] fp16 elegido — {100*(t32/t16-1):.0f}% más rápido, "
                          f"misma imagen.")
                else:
                    print("  [i] fp16 no acelera en este hardware: se mantiene float32.")
        except (RuntimeError, NotImplementedError) as exc:
            print(f"  [!] fp16 no soportado aquí ({exc}); se sigue en float32.")
            net.float()

    # Warm-up: la primera pasada incluye compilación de kernels y no es representativa.
    with torch.inference_mode():
        for _ in range(3):
            net(probe.to(device=device, dtype=dtype))
    sync(device)
    return device, dtype


def run_benchmark(net: torch.nn.Module, device: torch.device, args,
                  dtype: torch.dtype = torch.float32):
    """Mide el rendimiento sin necesidad de webcam. Útil para comparar máquinas."""
    n = args.benchmark
    gen = torch.Generator().manual_seed(0)
    probe = (torch.rand(1, args.input_nc, args.size, args.size, generator=gen) * 2 - 1
             ).to(device=device, dtype=dtype)

    times = []
    with torch.inference_mode():
        for _ in range(n):
            t0 = time.perf_counter()
            net(probe)
            sync(device)
            times.append(time.perf_counter() - t0)

    times.sort()
    mean = sum(times) / len(times)
    print(f"\n--- Benchmark ({n} forward passes de {args.size}x{args.size}) ---")
    print(f"Dispositivo : {describe_device(device)}")
    print(f"Precisión   : {'float16' if dtype == torch.float16 else 'float32'}")
    print(f"Media       : {mean * 1000:7.1f} ms  ->  {1 / mean:5.1f} fps")
    print(f"Mediana     : {times[len(times) // 2] * 1000:7.1f} ms")
    print(f"Mejor/peor  : {times[0] * 1000:7.1f} ms / {times[-1] * 1000:7.1f} ms")
    print("\n(Solo el generador; el bucle real añade captura, Canny y dibujado.)")


# --------------------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------------------

def resolve_checkpoint(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SystemExit(f"No existe el checkpoint: {path.resolve()}")
        return path
    for cand in DEFAULT_CKPT_CANDIDATES:
        if cand.is_file():
            return cand
    tried = "\n  ".join(str(c) for c in DEFAULT_CKPT_CANDIDATES)
    raise SystemExit(f"No encontré el generador. Rutas probadas:\n  {tried}\n"
                     "Pasa la ruta con --checkpoint.")


def strip_module_prefix(state_dict: OrderedDict) -> OrderedDict:
    """pix2pix guarda con net.module.state_dict() en GPU, pero otros flujos (o un
    torch.save del nn.DataParallel entero) dejan el prefijo 'module.'. Lo limpiamos."""
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    print("  [i] state_dict con prefijo 'module.' (DataParallel): limpiando claves.")
    return OrderedDict((k[len("module."):] if k.startswith("module.") else k, v)
                       for k, v in state_dict.items())


def apply_preset(args, ckpt_path: Path):
    """Aplica los ajustes Canny del modelo detectado, sin pisar lo que pidió el usuario.

    El nombre de la carpeta del checkpoint identifica el modelo (checkpoints/fire/...).
    Solo se rellenan los parámetros que NO aparecen en la línea de comandos, así que
    pasar --t1 explícitamente sigue teniendo prioridad.
    """
    preset = PRESETS.get(ckpt_path.parent.name)
    if not preset:
        return
    dado = set(sys.argv)
    aplicados = []
    for flag, clave in (("--t1", "t1"), ("--t2", "t2"),
                        ("--pre_blur", "pre_blur"), ("--pre_median", "pre_median")):
        if flag not in dado:
            setattr(args, clave, preset[clave])
            aplicados.append(f"{clave}={preset[clave]}")
    if aplicados:
        print(f"Preset '{preset['nombre']}'  : {'  '.join(aplicados)}")


class BancoDeModelos:
    """Carga generadores bajo demanda y los cachea, para que cambiar sea instantáneo.

    Cada unet_256 ocupa ~210 MB en float32, así que los cuatro caben de sobra en una GPU
    de 4 GB. Solo se paga la carga la primera vez que se pulsa cada tecla.
    """

    def __init__(self, args, device: torch.device, dtype: torch.dtype):
        self.args = args
        self.device = device
        self.dtype = dtype
        self.cache: dict[str, torch.nn.Module] = {}

    def disponible(self, modelo: dict) -> bool:
        return (CHECKPOINTS_DIR / modelo["carpeta"] / "latest_net_G.pth").is_file()

    def get(self, modelo: dict) -> torch.nn.Module | None:
        carpeta = modelo["carpeta"]
        if carpeta in self.cache:
            return self.cache[carpeta]
        ruta = CHECKPOINTS_DIR / carpeta / "latest_net_G.pth"
        if not ruta.is_file():
            print(f"[!] No existe {ruta}")
            return None
        print(f"Cargando {modelo['nombre']}...")
        net = build_generator(self.args)
        cargar_pesos(net, ruta, verbose=False)
        net.to(device=self.device, dtype=self.dtype)
        net.eval()
        self.cache[carpeta] = net
        return net


def build_generator(args) -> torch.nn.Module:
    return networks.define_G(
        args.input_nc, args.output_nc, args.ngf, args.netG,
        norm=args.norm, use_dropout=not args.no_dropout,
    )


def cargar_pesos(net: torch.nn.Module, ckpt_path: Path, verbose: bool = True):
    state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = strip_module_prefix(state_dict)
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"  [!] claves faltantes: {len(missing)} | inesperadas: {len(unexpected)}")
        for k in list(missing)[:5]:
            print(f"      falta      : {k}")
        for k in list(unexpected)[:5]:
            print(f"      inesperada : {k}")
        print("      Si son muchas, la arquitectura (--netG/--ngf/--norm) no coincide "
              "con la del entrenamiento.")
    elif verbose:
        print(f"  [ok] {len(state_dict)} tensores cargados sin discrepancias.")


def load_generator(args, device: torch.device, ckpt_path: Path | None = None) -> torch.nn.Module:
    if ckpt_path is None:
        ckpt_path = resolve_checkpoint(args.checkpoint)
    print(f"Cargando generador : {ckpt_path.resolve()}")

    net = build_generator(args)
    cargar_pesos(net, ckpt_path)
    net.to(device)
    net.eval()  # desactiva dropout y fija las estadísticas de BatchNorm
    return net


# --------------------------------------------------------------------------------------
# Preprocesado / postprocesado
# --------------------------------------------------------------------------------------

def make_edges(bgr: np.ndarray, st: dict) -> np.ndarray:
    """Idéntico a make_edges() de preprocess_canny.py. Devuelve BGR de 3 canales."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if st["pre_median"]:
        gray = cv2.medianBlur(gray, st["pre_median"])
    if st["pre_blur"]:
        gray = cv2.GaussianBlur(gray, (st["pre_blur"], st["pre_blur"]), 0)
    edges = cv2.Canny(gray, st["t1"], st["t2"],
                      apertureSize=st["aperture"], L2gradient=st["l2_gradient"])
    if st["invert"]:
        edges = cv2.bitwise_not(edges)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def to_tensor(bgr: np.ndarray, device: torch.device,
              dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """BGR uint8 HxWx3 -> tensor [1,3,H,W] en [-1,1].

    Replica ToTensor()+Normalize((0.5,)*3,(0.5,)*3) sobre una imagen RGB, que es lo que
    hace get_transform() en pix2pix/data/base_dataset.py.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0          # [0,1]
    arr = (arr - 0.5) / 0.5                        # [-1,1]
    chw = np.transpose(arr, (2, 0, 1))             # HWC -> CHW
    return torch.from_numpy(chw).unsqueeze(0).to(device=device, dtype=dtype)


def to_image(tensor: torch.Tensor) -> np.ndarray:
    """Tensor [1,3,H,W] en [-1,1] -> BGR uint8 HxWx3 (inverso de util.tensor2im).

    El .float() cubre también el caso float16, cuyo numpy no soporta bien.
    """
    arr = tensor[0].detach().cpu().float().numpy()
    arr = (np.transpose(arr, (1, 2, 0)) + 1.0) / 2.0 * 255.0
    rgb = np.clip(arr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------

FONT = cv2.FONT_HERSHEY_SIMPLEX


def build_display(fake: np.ndarray, edges: np.ndarray, st: dict, fps: float,
                  device: torch.device, modelo: dict, hud: int,
                  lado: int) -> np.ndarray:
    """Vista a pantalla completa: solo la alucinación, con un HUD opcional discreto.

    `hud`: 0 = nada (obra limpia), 1 = una línea mínima, 2 = ayuda completa.
    """
    # Escalado bicúbico a `lado` px; solo afecta a lo que se ve, no a la inferencia.
    vista = cv2.resize(fake, (lado, lado), interpolation=cv2.INTER_CUBIC)
    if hud == 0:
        return vista

    alto, ancho = vista.shape[:2]
    esc = max(0.45, lado / 1400.0)          # texto proporcional al tamaño de ventana

    def texto(img, s, x, y, escala, color, grosor=1):
        # Sombra para que el texto se lea sobre cualquier imagen de fondo.
        cv2.putText(img, s, (x + 1, y + 1), FONT, escala, (0, 0, 0), grosor + 1, cv2.LINE_AA)
        cv2.putText(img, s, (x, y), FONT, escala, color, grosor, cv2.LINE_AA)

    if hud == 1:
        texto(vista, modelo["nombre"], 14, int(30 * esc), 0.7 * esc, (255, 255, 255), 2)
        texto(vista, f"t1 {st['t1']}  t2 {st['t2']}  blur {st['pre_blur'] or '-'}  "
                     f"med {st['pre_median'] or '-'}",
              14, alto - int(14 * esc), 0.45 * esc, (185, 185, 185))
        return vista

    # hud == 2: ayuda completa sobre un velo oscuro
    velo = vista.copy()
    cv2.rectangle(velo, (0, 0), (ancho, alto), (0, 0, 0), -1)
    vista = cv2.addWeighted(vista, 0.35, velo, 0.65, 0)
    dens = 100.0 * (edges[:, :, 0] > 0).mean()
    y = int(40 * esc)
    paso = int(26 * esc)
    lineas = [
        (f"{modelo['nombre']}   ({modelo['pista']})", 0.7, (255, 255, 255)),
        (f"t1 {st['t1']}   t2 {st['t2']}   blur {st['pre_blur'] or 'off'}   "
         f"median {st['pre_median'] or 'off'}   bordes {dens:.1f}%", 0.5, (120, 230, 255)),
        (f"{device.type.upper()}   {fps:.0f} fps", 0.5, (150, 150, 150)),
        ("", 0.5, (0, 0, 0)),
        ("  ".join(f"[{m['tecla']}] {m['nombre']}" for m in MODELOS), 0.55, (255, 255, 255)),
        ("[+/-] ambos umbrales    [ [ ] ] t1    [ , . ] t2", 0.5, (200, 200, 200)),
        ("[b] blur    [m] median    [i] invertir    [f] espejo", 0.5, (200, 200, 200)),
        ("[r] restaurar preset    [s] captura    [h] este panel", 0.5, (200, 200, 200)),
        ("[q] salir", 0.5, (200, 200, 200)),
    ]
    for s, e, c in lineas:
        if s:
            texto(vista, s, int(24 * esc), y, e * esc, c)
        y += paso
    return vista


def cycle_kernel(current: int, forward: bool = True) -> int:
    idx = BLUR_CYCLE.index(current) if current in BLUR_CYCLE else 0
    return BLUR_CYCLE[(idx + (1 if forward else -1)) % len(BLUR_CYCLE)]


def clamp_thresholds(st: dict):
    """Canny exige 0 <= t1 <= t2. Mantenemos el invariante tras cada ajuste."""
    st["t1"] = int(np.clip(st["t1"], 0, 500))
    st["t2"] = int(np.clip(st["t2"], 0, 500))
    if st["t1"] > st["t2"]:
        st["t1"] = st["t2"]


# --------------------------------------------------------------------------------------

def main():
    args = parse_args()
    for name in ("pre_blur", "pre_median"):
        val = getattr(args, name)
        if val > 0 and val % 2 == 0:
            raise SystemExit(f"--{name} debe ser impar (recibido {val}).")

    print(f"torch {torch.__version__} | CUDA build: {torch.version.cuda or 'no'} | "
          f"MPS: {'sí' if mps_available() else 'no'} | {platform.system()}")
    device = pick_device(args)
    print(f"Dispositivo        : {describe_device(device)}")
    tune_backend(device)

    # El preset debe aplicarse antes de usar args.t1/t2 en el bucle.
    ckpt_path = resolve_checkpoint(args.checkpoint)
    apply_preset(args, ckpt_path)

    # Se carga en CPU y verify_device() lo mueve al acelerador solo si pasa la comprobación.
    net = load_generator(args, torch.device("cpu"), ckpt_path)
    dtype = torch.float32
    if not args.no_verify:
        device, dtype = verify_device(net, device, args)
    else:
        net.to(device)
        if args.half and device.type != "cpu":
            net.half()
            dtype = torch.float16
            print("  [!] --half sin verificar (--no_verify): no se comprueba si degrada.")
    if device.type == "cpu":
        print("  [i] En CPU espera pocos fps. Para tiempo real necesitas CUDA (NVIDIA) "
              "o MPS (Apple Silicon).")

    if args.benchmark:
        run_benchmark(net, device, args, dtype)
        return

    # Banco de modelos para cambiar en vivo con las teclas 1..4. El que ya está cargado
    # se registra en la caché para no leerlo dos veces del disco.
    banco = BancoDeModelos(args, device, dtype)
    modelo = PRESETS.get(ckpt_path.parent.name, MODELOS[0])
    banco.cache[modelo["carpeta"]] = net
    disponibles = [m for m in MODELOS if banco.disponible(m)]
    print("Modelos            : " +
          "  ".join(f"[{m['tecla']}] {m['nombre']}" for m in disponibles))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        msg = (f"No se pudo abrir la cámara {args.camera}. "
               "Prueba otro índice con --camera 1, o cierra la app que la esté usando.")
        if IS_MAC:
            msg += ("\n  En macOS, concede permiso en Ajustes del Sistema > Privacidad y "
                    "seguridad > Cámara a la terminal desde la que lanzas el script.")
        raise SystemExit(msg)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Cámara {args.camera}           : {real_w}x{real_h}")

    st = {
        "t1": args.t1, "t2": args.t2, "aperture": args.aperture,
        "l2_gradient": args.l2_gradient, "pre_blur": args.pre_blur,
        "pre_median": args.pre_median, "invert": args.invert,
        "mirror": not args.no_mirror,
    }
    defaults = dict(st)
    clamp_thresholds(st)

    screenshot_dir = Path(args.screenshot_dir)
    window = "Learning to See"
    # KEEPRATIO es imprescindible: la imagen es cuadrada y una pantalla 16:9 la estiraría.
    # Con esto se centra y aparecen bandas negras a los lados, sin deformar la obra.
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    if not args.ventana:
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    frame_times = deque(maxlen=30)
    n_shots = 0
    hud = args.hud            # 0 limpio | 1 mínimo | 2 ayuda

    print("\nPantalla completa. [1-4] modelo   [h] textos   [q] salir\n")
    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                print("[!] Se perdió el stream de la cámara.")
                break
            if st["mirror"]:
                frame = cv2.flip(frame, 1)

            # Mismo estirado a size x size que en entrenamiento (sin preservar aspect ratio).
            frame = cv2.resize(frame, (args.size, args.size), interpolation=cv2.INTER_AREA)
            edges = make_edges(frame, st)

            with torch.inference_mode():
                out = net(to_tensor(edges, device, dtype))
            sync(device)  # los fps medidos deben incluir el trabajo real de la GPU
            fake = to_image(out)

            frame_times.append(time.perf_counter() - t0)
            fps = len(frame_times) / max(sum(frame_times), 1e-6)

            vista = build_display(fake, edges, st, fps, device, modelo, hud,
                                  args.display_size)
            cv2.imshow(window, vista)

            key = cv2.waitKey(1) & 0xFF
            if key == 255:                                   # sin tecla
                pass
            elif key in (ord("q"), 27):                      # q / ESC
                break
            elif key in TECLAS:                              # 1..4: cambiar de modelo
                elegido = TECLAS[key]
                nuevo = banco.get(elegido)
                if nuevo is not None:
                    net, modelo = nuevo, elegido
                    # Cada modelo trae su propio filtrado calibrado.
                    st.update(t1=elegido["t1"], t2=elegido["t2"],
                              pre_blur=elegido["pre_blur"],
                              pre_median=elegido["pre_median"])
                    defaults = dict(st)
                    clamp_thresholds(st)
                    print(f"-> {elegido['nombre']}  t1={st['t1']} t2={st['t2']} "
                          f"blur={st['pre_blur'] or 'off'} median={st['pre_median'] or 'off'}"
                          f"   (probar con: {elegido['pista']})")
            elif key == ord("h"):                            # ciclar textos
                hud = (hud + 1) % 3
            elif key in (ord("+"), ord("=")):                # ambos umbrales arriba
                st["t1"] += args.step
                st["t2"] += args.step
                clamp_thresholds(st)
            elif key in (ord("-"), ord("_")):                # ambos umbrales abajo
                st["t1"] -= args.step
                st["t2"] -= args.step
                clamp_thresholds(st)
            elif key == ord("["):
                st["t1"] -= args.step
                clamp_thresholds(st)
            elif key == ord("]"):
                st["t1"] += args.step
                clamp_thresholds(st)
            elif key == ord(","):
                st["t2"] -= args.step
                clamp_thresholds(st)
            elif key == ord("."):
                st["t2"] += args.step
                clamp_thresholds(st)
            elif key == ord("b"):
                st["pre_blur"] = cycle_kernel(st["pre_blur"])
            elif key == ord("m"):
                st["pre_median"] = cycle_kernel(st["pre_median"])
            elif key == ord("i"):
                st["invert"] = not st["invert"]
            elif key == ord("f"):
                st["mirror"] = not st["mirror"]
            elif key == ord("r"):
                st.update(defaults)
                print("  [i] parámetros restaurados al preset del modelo.")
            elif key == ord("s"):
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                # Se guarda la obra limpia, sin textos ni escalado de pantalla.
                # imencode+tofile para tolerar rutas con acentos en Windows.
                out_path = screenshot_dir / f"{stamp}_{modelo['carpeta']}.png"
                enc_ok, buf = cv2.imencode(".png", fake)
                if enc_ok:
                    buf.tofile(str(out_path))
                    n_shots += 1
                    print(f"  [s] captura guardada: {out_path.name}")

            # La ventana cerrada con la X también termina el bucle.
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        print("\nInterrumpido.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nFin. Umbrales finales: --t1 {st['t1']} --t2 {st['t2']} "
          f"--pre_blur {st['pre_blur']} --pre_median {st['pre_median']}"
          f"{' --invert' if st['invert'] else ''}")
    if n_shots:
        print(f"Capturas guardadas: {n_shots} en {screenshot_dir.resolve()}")


if __name__ == "__main__":
    main()

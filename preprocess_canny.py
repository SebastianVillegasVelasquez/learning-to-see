"""
preprocess_canny.py — genera el dataset pareado (A|B) estilo "Learning to See" (Memo Akten).

Patrón de Akten (webcam-pix2pix-tensorflow / cci-auto-pix2pix):
un solo dominio de imágenes (aquí: océano/olas). Para cada imagen del dominio:

    B = la imagen real (target)
    A = la misma imagen pasada por cv2.Canny (input, "bordes")

El par se guarda concatenado horizontalmente [A | B] en un único PNG de 512x256,
que es el formato que consume pytorch-CycleGAN-and-pix2pix con
`--dataset_mode aligned --direction AtoB`.

Uso típico:
    python preprocess_canny.py
    python preprocess_canny.py --t1 60 --t2 180 --pre_blur 3 --pre_median 3
    python preprocess_canny.py --preview 8      # inspeccionar antes de procesar todo
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Genera pares A|B (Canny | original) para pix2pix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_dir", default="./data/raw_images",
                   help="Carpeta con las imágenes crudas del dominio.")
    p.add_argument("--output_dir", default="./data/AB",
                   help="Raíz del dataset pix2pix (se crean subcarpetas train/val/test).")
    p.add_argument("--size", type=int, default=256,
                   help="Lado del recuadro final. Se estira sin preservar aspect ratio (igual que Akten).")

    # --- parámetros Canny ---
    p.add_argument("--t1", type=int, default=100, help="Umbral bajo de cv2.Canny.")
    p.add_argument("--t2", type=int, default=200, help="Umbral alto de cv2.Canny.")
    p.add_argument("--aperture", type=int, default=3, choices=[3, 5, 7],
                   help="Tamaño del kernel Sobel de Canny.")
    p.add_argument("--l2_gradient", action="store_true",
                   help="Usa norma L2 en Canny (más preciso, algo más lento).")

    # --- pre-filtrado anti-ruido (como en el webcam-pix2pix de Akten) ---
    p.add_argument("--pre_blur", type=int, default=0,
                   help="Kernel de GaussianBlur antes del Canny. 0 = desactivado. Debe ser impar.")
    p.add_argument("--pre_median", type=int, default=0,
                   help="Kernel de medianBlur antes del Canny. 0 = desactivado. Debe ser impar.")

    # --- salida ---
    p.add_argument("--invert", action="store_true",
                   help="Bordes negros sobre fondo blanco en vez de blanco sobre negro.")
    p.add_argument("--split", default="train",
                   help="Subcarpeta destino dentro de output_dir (train/val/test).")
    p.add_argument("--val_frac", type=float, default=0.0,
                   help="Fracción de imágenes desviadas a la carpeta 'val' (0 = ninguna).")
    p.add_argument("--seed", type=int, default=0, help="Semilla para el reparto train/val.")
    p.add_argument("--preview", type=int, default=0,
                   help="Si >0, procesa solo N imágenes hacia ./data/preview para calibrar t1/t2.")
    p.add_argument("--overwrite", action="store_true",
                   help="Reprocesa imágenes cuyo par ya existe en destino.")
    p.add_argument("--min_edge_density", type=float, default=0.0, metavar="PCT",
                   help="Descarta imágenes cuyo lado A tenga menos de este %% de píxeles de "
                        "borde. Sirve para sacar del dataset las tomas planas o sin textura "
                        "(agua lisa, cielo vacío) que enseñarían al modelo a devolver "
                        "imágenes vacías. 0 = sin filtro.")
    p.add_argument("--max_edge_density", type=float, default=100.0, metavar="PCT",
                   help="Descarta imágenes por encima de este %% de bordes (ruido excesivo).")
    return p.parse_args()


def odd_or_zero(value: int, name: str) -> int:
    """Los kernels de OpenCV deben ser impares y positivos; 0 significa 'desactivado'."""
    if value <= 0:
        return 0
    if value % 2 == 0:
        raise SystemExit(f"--{name} debe ser impar (recibido {value}).")
    return value


def make_edges(bgr: np.ndarray, args) -> np.ndarray:
    """Devuelve la versión 'A' (bordes) en BGR de 3 canales."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if args.pre_median:
        gray = cv2.medianBlur(gray, args.pre_median)
    if args.pre_blur:
        gray = cv2.GaussianBlur(gray, (args.pre_blur, args.pre_blur), 0)

    edges = cv2.Canny(gray, args.t1, args.t2,
                      apertureSize=args.aperture, L2gradient=args.l2_gradient)
    if args.invert:
        edges = cv2.bitwise_not(edges)

    # pix2pix espera 3 canales en ambos lados del par.
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def list_images(input_dir: Path):
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)


def main():
    args = parse_args()
    args.pre_blur = odd_or_zero(args.pre_blur, "pre_blur")
    args.pre_median = odd_or_zero(args.pre_median, "pre_median")

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"No existe la carpeta de entrada: {input_dir.resolve()}")

    paths = list_images(input_dir)
    if not paths:
        raise SystemExit(f"No se encontraron imágenes en {input_dir.resolve()}. "
                         f"Extensiones aceptadas: {', '.join(sorted(IMG_EXTS))}")

    if args.preview:
        paths = paths[:args.preview]
        dest_of = {p: Path("./data/preview") for p in paths}
    else:
        rng = random.Random(args.seed)
        val_set = set()
        if args.val_frac > 0:
            shuffled = paths[:]
            rng.shuffle(shuffled)
            n_val = int(round(len(paths) * args.val_frac))
            val_set = set(shuffled[:n_val])
        out_root = Path(args.output_dir)
        dest_of = {p: out_root / ("val" if p in val_set else args.split) for p in paths}

    for d in set(dest_of.values()):
        d.mkdir(parents=True, exist_ok=True)

    written = skipped = failed = filtered = 0
    densities = []
    for i, src in enumerate(paths, 1):
        out_path = dest_of[src] / f"{src.stem}.png"
        if out_path.exists() and not args.overwrite and not args.preview:
            skipped += 1
            continue

        # imdecode en vez de imread: soporta rutas con acentos/no-ASCII en Windows.
        try:
            buf = np.fromfile(str(src), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except OSError:
            img = None
        if img is None:
            print(f"  [!] ilegible, se omite: {src}", file=sys.stderr)
            failed += 1
            continue

        # Estirado directo a size x size, sin preservar aspect ratio (igual que Akten).
        b = cv2.resize(img, (args.size, args.size), interpolation=cv2.INTER_AREA)
        a = make_edges(b, args)

        # Filtro de densidad: una entrada casi vacía enseña al modelo a producir vacío,
        # y una saturada de ruido no tiene estructura que aprender.
        density = 100.0 * (a[:, :, 0] > 0).mean()
        if density < args.min_edge_density or density > args.max_edge_density:
            filtered += 1
            continue

        ab = np.concatenate([a, b], axis=1)  # [A | B] -> size*2 x size

        ok, encoded = cv2.imencode(".png", ab)
        if not ok:
            print(f"  [!] fallo al codificar: {src}", file=sys.stderr)
            failed += 1
            continue
        encoded.tofile(str(out_path))
        written += 1
        densities.append(density)

        if i % 50 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)} procesadas...")

    print("\n--- Resumen ---")
    print(f"Entrada        : {input_dir.resolve()}")
    for d in sorted(set(dest_of.values())):
        print(f"Salida         : {d.resolve()}")
    print(f"Tamaño par     : {args.size * 2}x{args.size} (A|B)")
    print(f"Canny          : t1={args.t1} t2={args.t2} aperture={args.aperture} "
          f"L2={args.l2_gradient} invert={args.invert}")
    print(f"Pre-filtro     : blur={args.pre_blur or 'off'} median={args.pre_median or 'off'}")
    print(f"Escritas       : {written}")
    print(f"Omitidas (ya)  : {skipped}")
    print(f"Fallidas       : {failed}")
    if args.min_edge_density > 0 or args.max_edge_density < 100:
        print(f"Filtradas      : {filtered} (fuera de "
              f"{args.min_edge_density}-{args.max_edge_density}% de bordes)")
    if densities:
        d = np.array(densities)
        print(f"Densidad bordes: media {d.mean():.1f}%  p10 {np.percentile(d, 10):.1f}%  "
              f"p90 {np.percentile(d, 90):.1f}%")
        print(f"                 -> para inferencia: --density_range "
              f"{np.percentile(d, 10):.1f} {np.percentile(d, 90):.1f}")
    if args.preview:
        print("\nModo preview: revisa ./data/preview y ajusta --t1/--t2/--pre_blur antes "
              "de procesar el dataset completo.")


if __name__ == "__main__":
    main()

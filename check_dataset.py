"""
check_dataset.py — valida ./data/raw_images antes de preprocesar / subir a Colab.

Comprueba que cada archivo se decodifica realmente (no basta con la extensión),
reporta el total, resoluciones y avisa de imágenes pequeñas, grises o duplicadas por nombre.

Uso:
    python check_dataset.py
    python check_dataset.py --input_dir ./data/raw_images --min_size 256
    python check_dataset.py --delete_broken     # borra los archivos ilegibles
"""

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Valida el dataset crudo de imágenes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_dir", default="./data/raw_images")
    p.add_argument("--min_size", type=int, default=256,
                   help="Avisa si el lado menor de la imagen es inferior a este valor.")
    p.add_argument("--delete_broken", action="store_true",
                   help="Elimina los archivos que no se pueden decodificar.")
    return p.parse_args()


def human(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"No existe la carpeta: {input_dir.resolve()}")

    all_files = [p for p in input_dir.rglob("*") if p.is_file()]
    candidates = [p for p in all_files if p.suffix.lower() in IMG_EXTS]
    others = [p for p in all_files if p.suffix.lower() not in IMG_EXTS]

    ok, broken, small, grayscale = [], [], [], []
    sizes, total_bytes = Counter(), 0

    for path in candidates:
        try:
            buf = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except OSError:
            img = None
        if img is None:
            broken.append(path)
            continue

        h, w = img.shape[:2]
        ok.append(path)
        sizes[(w, h)] += 1
        total_bytes += path.stat().st_size
        if min(h, w) < args.min_size:
            small.append((path, w, h))
        # Si los 3 canales son idénticos, la imagen es gris disfrazada de color.
        if np.array_equal(img[:, :, 0], img[:, :, 1]) and np.array_equal(img[:, :, 1], img[:, :, 2]):
            grayscale.append(path)

    dup_names = [name for name, c in Counter(p.stem for p in ok).items() if c > 1]

    print(f"Carpeta          : {input_dir.resolve()}")
    print(f"Archivos totales : {len(all_files)}")
    print(f"IMÁGENES VÁLIDAS : {len(ok)}")
    print(f"Ilegibles        : {len(broken)}")
    print(f"No-imagen        : {len(others)}")
    if ok:
        print(f"Peso total       : {human(total_bytes)}  (media {human(total_bytes / len(ok))})")
        print(f"Resoluciones     : {len(sizes)} distintas; top 5:")
        for (w, h), c in sizes.most_common(5):
            print(f"                   {w}x{h}  ->  {c}")

    if broken:
        print(f"\n[!] {len(broken)} archivo(s) ilegible(s):")
        for p in broken[:20]:
            print(f"    {p}")
        if len(broken) > 20:
            print(f"    ... y {len(broken) - 20} más")
        if args.delete_broken:
            for p in broken:
                p.unlink()
            print(f"    -> eliminados {len(broken)} archivos.")
        else:
            print("    Usa --delete_broken para eliminarlos.")

    if small:
        print(f"\n[!] {len(small)} imagen(es) con lado menor a {args.min_size}px "
              f"(se escalarán hacia arriba y perderán nitidez):")
        for p, w, h in small[:10]:
            print(f"    {w}x{h}  {p.name}")
        if len(small) > 10:
            print(f"    ... y {len(small) - 10} más")

    if grayscale:
        print(f"\n[!] {len(grayscale)} imagen(es) en escala de grises "
              f"(el modelo aprenderá a generar B&N para esos casos).")

    if others:
        print(f"\n[!] {len(others)} archivo(s) sin extensión de imagen (se ignorarán):")
        for p in others[:10]:
            print(f"    {p.name}")

    if dup_names:
        print(f"\n[!] {len(dup_names)} nombre(s) base duplicado(s) en subcarpetas distintas; "
              f"los pares A|B se sobrescribirían entre sí: {', '.join(dup_names[:10])}")

    print("\n--- Veredicto ---")
    if not ok:
        print("VACÍO: no hay imágenes utilizables. Añade imágenes a raw_images.")
    elif len(ok) < 200:
        print(f"{len(ok)} imágenes: suficiente para una prueba, pero pix2pix suele necesitar "
              f"~400+ para resultados decentes en un dominio.")
    else:
        print(f"{len(ok)} imágenes: listo para `python preprocess_canny.py`.")


if __name__ == "__main__":
    main()

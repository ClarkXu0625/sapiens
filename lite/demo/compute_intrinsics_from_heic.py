#!/usr/bin/env python3
"""
Compute pinhole intrinsics K for an iPhone HEIC without any command-line flags.

Usage:
  python compute_K_no_flags.py /path/to/IMG_XXXX.HEIC

Outputs:
  - <image>.K_raw.npy / .json        (K for stored raster W×H)
  - <image>.K_rotated.npy / .json    (K for auto-rotated raster if Orientation is 6/8)
"""

import json
import math
import os
import subprocess
import sys
from typing import Dict, Any, Tuple

import numpy as np


def exif_json_numeric(path: str) -> Dict[str, Any]:
    keys = [
        "ImageWidth", "ImageHeight", "Orientation",
        "FocalLength", "FocalLengthIn35mmFormat", "FieldOfView"
    ]
    cmd = ["exiftool", "-json", "-n"] + [f"-{k}" for k in keys] + [path]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        sys.stderr.write("Failed to run exiftool:\n" + e.output.decode("utf-8", errors="ignore"))
        sys.exit(1)
    data = json.loads(out.decode("utf-8", errors="ignore"))
    if not data:
        sys.stderr.write("No EXIF data returned.\n")
        sys.exit(1)
    rec = data[0]
    # Normalize missing numeric fields to None
    for k in keys:
        if k not in rec:
            rec[k] = None
    return rec


def compute_from_35mm(W: int, H: int, f35: float) -> Tuple[float, float, float, float]:
    fx = W * float(f35) / 36.0
    fy = H * float(f35) / 24.0
    cx, cy = W / 2.0, H / 2.0
    return fx, fy, cx, cy


def compute_from_fov(W: int, H: int, fovx_deg: float) -> Tuple[float, float, float, float]:
    # Horizontal FOV assumed; EXIF FieldOfView is typically horizontal for iPhone stills.
    fovx = math.radians(float(fovx_deg))
    fx = (W / 2.0) / math.tan(fovx / 2.0)
    # Assume square pixels; scale fy by aspect ratio
    fy = fx * (H / W)
    cx, cy = W / 2.0, H / 2.0
    return fx, fy, cx, cy


def build_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def save_sidecars(base: str, tag: str, K: np.ndarray, meta: Dict[str, Any]):
    np.save(f"{base}.K_{tag}.npy", K)
    info = {
        "tag": tag,
        "width": meta["W"],
        "height": meta["H"],
        "orientation_tag": meta["Orientation"],
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "method": meta["method"],
        "notes": [
            "Principal point assumed at image center.",
            "Square pixels assumed.",
            "If you resize/crop later, scale/shift K accordingly.",
            "Use *_rotated if your loader auto-applies EXIF orientation (6/8)."
        ]
    }
    with open(f"{base}.K_{tag}.json", "w") as f:
        json.dump(info, f, indent=2)


def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python compute_K_no_flags.py /path/to/IMG_XXXX.HEIC\n")
        sys.exit(2)

    path = sys.argv[1]
    if not os.path.isfile(path):
        sys.stderr.write(f"File not found: {path}\n")
        sys.exit(2)

    ex = exif_json_numeric(path)

    W = int(ex.get("ImageWidth") or 0)
    H = int(ex.get("ImageHeight") or 0)
    if W <= 0 or H <= 0:
        sys.stderr.write("Missing ImageWidth/ImageHeight in EXIF.\n")
        sys.exit(1)

    orientation = int(ex.get("Orientation") or 1)
    f35 = ex.get("FocalLengthIn35mmFormat")
    fovx = ex.get("FieldOfView")

    # Prefer 35mm-equiv; fallback to FieldOfView
    method = None
    if f35 is not None:
        method = "35mm-equiv"
        fx, fy, cx, cy = compute_from_35mm(W, H, float(f35))
    elif fovx is not None:
        method = "FieldOfView"
        fx, fy, cx, cy = compute_from_fov(W, H, float(fovx))
    else:
        sys.stderr.write(
            "No FocalLengthIn35mmFormat or FieldOfView in EXIF.\n"
            "Capture with an ARKit/LiDAR app to get intrinsics directly.\n"
        )
        sys.exit(1)

    # K for the raw stored raster (no orientation applied)
    K_raw = build_K(fx, fy, cx, cy)

    # K for the auto-rotated raster (swap W/H) only if 90°/270° orientation
    if orientation in (6, 8):
        # Recompute with swapped dimensions
        W_r, H_r = H, W
        if method == "35mm-equiv":
            fx_r, fy_r, cx_r, cy_r = compute_from_35mm(W_r, H_r, float(f35))
        else:
            fx_r, fy_r, cx_r, cy_r = compute_from_fov(W_r, H_r, float(fovx))
        K_rot = build_K(fx_r, fy_r, cx_r, cy_r)
    else:
        # Same as raw when orientation is normal/180
        K_rot = K_raw.copy()

    base = os.path.splitext(path)[0]

    save_sidecars(base, "raw", K_raw, {
        "W": W, "H": H, "Orientation": orientation, "method": method
    })
    save_sidecars(base, "rotated", K_rot, {
        "W": (H if orientation in (6, 8) else W),
        "H": (W if orientation in (6, 8) else H),
        "Orientation": orientation, "method": method
    })

    print("=== EXIF ===")
    print(f"Size: {W}x{H}, Orientation: {orientation}, Method: {method}, "
          f"FocalLengthIn35mmFormat: {f35}, FieldOfView: {fovx}")
    print("\nK_raw =\n", K_raw)
    print("\nK_rotated =\n", K_rot)
    print(f"\nSaved: {base+'.K_raw.npy'}, {base+'.K_raw.json'}, "
          f"{base+'.K_rotated.npy'}, {base+'.K_rotated.json'}")


if __name__ == "__main__":
    main()

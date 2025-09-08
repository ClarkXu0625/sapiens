import argparse, os
import numpy as np
import cv2
import open3d as o3d
import pdb

def build_intrinsics(W, H, f=None, fov_deg=None):
    """
    Create K with principal point at center and fx=fy=f.
    If f is None, derive from horizontal FOV (default 60°).
    """
    if f is None:
        if fov_deg is None:
            fov_deg = 60.0
        f = 0.5 * W / np.tan(0.5 * np.deg2rad(fov_deg))
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float32)
    return K

def map_relative_to_metric(depth_rel, mask=None, invert=False,
                           z_near=None, z_far=None,
                           median_m=None, range_m=1.5):
    """
    Map relative (arbitrary) depth to metric meters.
    Options:
      - explicit window [z_near, z_far]
      - or target median +/- range/2
    """
    if mask is None:
        mask = np.ones_like(depth_rel, dtype=bool)
    if mask.ndim == 3:
        mask = mask[..., 0] > 0
    else:
        mask = mask > 0

    finite = np.isfinite(depth_rel)
    valid = finite & mask
    if not np.any(valid):
        raise ValueError("No valid pixels in depth/mask.")

    dmin = float(np.nanmin(depth_rel[valid]))
    dmax = float(np.nanmax(depth_rel[valid]))
    denom = (dmax - dmin) if (dmax > dmin) else 1.0

    dnorm = (depth_rel - dmin) / (denom + 1e-8)
    #pdb.set_trace()
    # dmin, dmax = depth_rel.min(), depth_rel.max()
    # dnorm = (depth_rel - dmin) / (dmax - dmin + 1e-8)

    if invert:
        dnorm = 1.0 - dnorm

    if (z_near is not None) and (z_far is not None):
        Z = z_near + dnorm * (z_far - z_near)
    else:
        if median_m is None:
            median_m = 1.5
        z_near_guess = max(0.05, median_m - range_m * 0.5)
        z_far_guess  = median_m + range_m * 0.5
        Z = z_near_guess + dnorm * (z_far_guess - z_near_guess)

    Z[~valid] = np.nan

    # Debug stats
    z_valid = Z[np.isfinite(Z)]
    print(f"[MapDepth] rel: dmin={dmin:.4f}, dmax={dmax:.4f} "
          f"-> metric: z_min={np.min(z_valid):.3f} m, z_med={np.median(z_valid):.3f} m, z_max={np.max(z_valid):.3f} m")
    return Z, valid

def backproject_to_camera(Z, K):
    """
    Backproject (u,v,Z) -> (X,Y,Z) in camera coords (meters).
    X right, Y down, Z forward.
    """
    H, W = Z.shape
    fx, fy = float(K[0,0]), float(K[1,1])
    cx, cy = float(K[0,2]), float(K[1,2])

    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    u, v = np.meshgrid(xs, ys)          # HxW

    Zf = Z.reshape(-1)
    uf = u.reshape(-1)
    vf = v.reshape(-1)
    good = np.isfinite(Zf) & (Zf > 0)

    Zf = Zf[good]
    uf = uf[good]
    vf = vf[good]

    X = (uf - cx) / fx * Zf
    Y = (vf - cy) / fy * Zf
    pts = np.stack([X, Y, Zf], axis=1)  # Nx3
    return pts, good.reshape(H, W)

def extract_colors(rgb_path, target_shape, valid_mask_flat):
    if (rgb_path is None) or (not os.path.exists(rgb_path)):
        return None
    img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        return None
    if img.shape[:2] != target_shape:
        img = cv2.resize(img, (target_shape[1], target_shape[0]),
                         interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = (rgb.reshape(-1, 3).astype(np.float32) / 255.0)[valid_mask_flat]
    return rgb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth_npy", required=True, help="Sapiens depth .npy (HxW float32 relative)")
    ap.add_argument("--rgb", required=False, help="Aligned RGB image (for colors)")
    ap.add_argument("--mask_npy", required=False, help="Foreground mask .npy (HxW or HxWx1)")
    ap.add_argument("--ply_out", required=True, help="Output PLY path")
    # intrinsics
    ap.add_argument("--f", type=float, default=None, help="Focal length in pixels (fx=fy=f)")
    ap.add_argument("--fov_deg", type=float, default=None, help="Horizontal FOV in degrees (alt to --f)")
    # depth mapping
    ap.add_argument("--invert_depth", action="store_true", help="Invert relative depth")
    ap.add_argument("--z_near_m", type=float, default=None, help="Near depth in meters")
    ap.add_argument("--z_far_m", type=float, default=None, help="Far  depth in meters")
    ap.add_argument("--median_m", type=float, default=None, help="Target median depth (meters)")
    ap.add_argument("--range_m", type=float, default=2.0, help="Span around median (meters)")
    # cloud options
    ap.add_argument("--voxel_down_m", type=float, default=0.0, help="Optional voxel size (meters)")
    args = ap.parse_args()

    depth_rel = np.load(args.depth_npy).astype(np.float32)
    if depth_rel.ndim != 2:
        raise ValueError("depth_npy must be HxW float array.")
    H, W = depth_rel.shape
    print(f"[Depth] shape={depth_rel.shape}, finite={np.isfinite(depth_rel).sum()}/{depth_rel.size}")

    # align mask
    mask = None
    if args.mask_npy and os.path.exists(args.mask_npy):
        mask = np.load(args.mask_npy)
        if mask.ndim == 3:
            mask = mask[..., 0]
        # Resize mask to depth resolution (nearest) to avoid skewed stats
        if mask.shape[:2] != depth_rel.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(bool)
        print(f"[Mask]  shape={mask.shape}, true={(mask>0).sum()}")

    # Intrinsics
    K = build_intrinsics(W, H, f=args.f, fov_deg=args.fov_deg)
    print("[Intrinsics]\n", K)

    # Relative -> metric depth
    Z, valid_for_stats = map_relative_to_metric(
        depth_rel, mask=mask, invert=args.invert_depth,
        z_near=args.z_near_m, z_far=args.z_far_m,
        median_m=args.median_m, range_m=args.range_m
    )

    # Backproject
    pts, valid_mask = backproject_to_camera(Z, K)
    print(f"[Points] {pts.shape[0]} valid points")

    # Colors 
    colors = extract_colors(args.rgb, target_shape=(H, W), valid_mask_flat=valid_mask.reshape(-1))

    # Open3D PCD
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    if args.voxel_down_m and args.voxel_down_m > 0:
        pcd = pcd.voxel_down_sample(args.voxel_down_m)

    os.makedirs(os.path.dirname(args.ply_out), exist_ok=True)

    ok = o3d.io.write_point_cloud(args.ply_out, pcd,
                                  write_ascii=False, compressed=False, print_progress=True)
    if not ok:
        raise RuntimeError("Failed to write PLY.")
    print(f"[OK] Saved: {args.ply_out}")

if __name__ == "__main__":
    main()

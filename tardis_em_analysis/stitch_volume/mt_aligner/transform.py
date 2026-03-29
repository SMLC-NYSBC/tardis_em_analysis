"""
Fit a rigid 2-D transformation (rotation + translation + optional isotropic
scale) from matched point pairs via SVD-based Procrustes analysis.
"""

import numpy as np


def fit_rigid_transform_2d(ref_xy, mov_xy, allow_scale=True):
    """
    Compute the rigid 2-D transform that maps *mov_xy* onto *ref_xy*.

    Args:
        ref_xy:      [m, 2]  reference (fixed) XY positions.
        mov_xy:      [m, 2]  moving XY positions.
        allow_scale: Estimate an isotropic scale factor (needs >= 3 pairs).

    Returns:
        angle_deg, tx, ty, scale
    """
    assert len(ref_xy) >= 2, f"Need >= 2 matched pairs, got {len(ref_xy)}"

    ref_c = ref_xy - ref_xy.mean(axis=0)
    mov_c = mov_xy - mov_xy.mean(axis=0)

    # Cross-covariance  →  SVD
    H = mov_c.T @ ref_c
    U, S, Vt = np.linalg.svd(H)

    # Proper rotation (det = +1)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, np.sign(d)]) @ U.T

    angle_deg = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

    # Isotropic scale
    if allow_scale and len(ref_xy) >= 3:
        mov_var = np.sum(mov_c ** 2)
        scale = np.sqrt(np.sum(ref_c ** 2) / mov_var) if mov_var > 1e-8 else 1.0
    else:
        scale = 1.0

    # Translation: ref_mean = scale * R @ mov_mean + t
    t = ref_xy.mean(axis=0) - scale * (R @ mov_xy.mean(axis=0))

    return angle_deg, float(t[0]), float(t[1]), float(scale)

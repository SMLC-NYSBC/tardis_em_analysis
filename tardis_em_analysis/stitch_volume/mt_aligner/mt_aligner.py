"""
Compute a rigid 2-D transform between consecutive serial-section tomograms
by matching microtubule endpoints near the section boundary.

Algorithm (adapted from Weber et al., PLoS ONE 2014):
    1.  Extract MT endpoints that reach the boundary zone of each section.
    2.  Score every candidate pair by XY distance + direction angle.
    3.  Solve the optimal one-to-one assignment (Hungarian).
    4.  Fit a rigid transform (rotation + translation + scale) via SVD Procrustes.
    5.  Iteratively refine: apply transform → re-match → re-fit.
"""

import logging

import numpy as np

from tardis_em_analysis.stitch_volume.mt_aligner.endpoints import (
    extract_boundary_endpoints,
)
from tardis_em_analysis.stitch_volume.mt_aligner.matching import match_endpoints
from tardis_em_analysis.stitch_volume.mt_aligner.transform import (
    fit_rigid_transform_2d,
)

logger = logging.getLogger("tardis_em")


def _apply_xy_transform(endpoints, angle_deg, tx, ty, scale):
    """Return a *copy* of *endpoints* with pos[:2] rigidly transformed."""
    theta = np.deg2rad(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    out = []
    for ep in endpoints:
        ep2 = {**ep, "pos": ep["pos"].copy(), "dir": ep["dir"].copy()}
        xy = ep2["pos"][:2]
        ep2["pos"][:2] = scale * (R @ xy) + np.array([tx, ty])
        ep2["dir"][:2] = R @ ep2["dir"][:2]
        out.append(ep2)
    return out


def compute_mt_transform(
    coords_fixed,
    coords_moving,
    max_xy_dist=500.0,
    max_angle_deg=30.0,
    z_band_fraction=0.15,
    refine_iters=3,
):
    """
    Compute the rigid 2-D transform that aligns *coords_moving* onto
    *coords_fixed* using microtubule endpoint matching.

    Args:
        coords_fixed:    [n, 4]  coordinates of the fixed (reference) section.
        coords_moving:   [n, 4]  coordinates of the moving section.
        max_xy_dist:     Maximum XY distance (pixels) for a valid pair.
        max_angle_deg:   Maximum direction-angle difference (degrees).
        z_band_fraction: Fraction of Z-range that defines the boundary zone.
        refine_iters:    Number of match → fit refinement cycles.

    Returns:
        dict  {Angle, Tx, Ty, Scale, Score, n_matches}
    """
    identity = {"Angle": 0.0, "Tx": 0.0, "Ty": 0.0, "Scale": 1.0,
                "Score": 0.0, "n_matches": 0}

    if coords_fixed is None or coords_moving is None:
        return identity
    if len(coords_fixed) == 0 or len(coords_moving) == 0:
        return identity

    # 1. Extract boundary endpoints
    #    bottom of fixed (high-Z)  ↔  top of moving (low-Z)
    ref_eps = extract_boundary_endpoints(
        coords_fixed, boundary="bottom", z_band_fraction=z_band_fraction
    )
    mov_eps = extract_boundary_endpoints(
        coords_moving, boundary="top", z_band_fraction=z_band_fraction
    )

    if len(ref_eps) < 2 or len(mov_eps) < 2:
        logger.warning("Too few boundary endpoints for MT matching "
                       f"(ref={len(ref_eps)}, mov={len(mov_eps)})")
        return identity

    # Accumulated transform
    total_angle, total_tx, total_ty, total_scale = 0.0, 0.0, 0.0, 1.0
    current_mov = mov_eps

    for it in range(max(1, refine_iters)):
        matches, ref_xy, mov_xy = match_endpoints(
            ref_eps, current_mov, max_xy_dist, max_angle_deg
        )

        if len(matches) < 2:
            if it == 0:
                logger.warning(f"MT matching found only {len(matches)} pairs")
                return identity
            break  # keep last good fit

        angle, tx, ty, sc = fit_rigid_transform_2d(ref_xy, mov_xy,
                                                    allow_scale=(it == 0))

        total_angle += angle
        total_tx += tx
        total_ty += ty
        total_scale *= sc

        # Apply incremental transform for next iteration's matching
        current_mov = _apply_xy_transform(current_mov, angle, tx, ty, sc)

        logger.debug(f"  MT refine iter {it}: {len(matches)} matches, "
                     f"angle={angle:.3f}, tx={tx:.1f}, ty={ty:.1f}, sc={sc:.4f}")

    n_matched = len(matches)
    score = n_matched / max(len(ref_eps), len(mov_eps))

    logger.info(f"MT matching: {n_matched}/{max(len(ref_eps), len(mov_eps))} "
                f"endpoints matched  (score={score:.3f})")

    return {
        "Angle": total_angle,
        "Tx": total_tx,
        "Ty": total_ty,
        "Scale": total_scale,
        "Score": score,
        "n_matches": n_matched,
    }

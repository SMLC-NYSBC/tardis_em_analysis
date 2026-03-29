"""
Score and match microtubule endpoints between consecutive sections.

Uses a combined distance + direction score with the Hungarian algorithm
for globally optimal one-to-one assignment (cf. Weber et al., 2014).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def _pair_score(pos1, dir1, pos2, dir2, max_xy_dist, max_angle_deg,
                w_dist=0.7, w_angle=0.3):
    """Cost for a candidate pair.  Returns np.inf if thresholds are exceeded."""
    xy_dist = np.linalg.norm(pos1[:2] - pos2[:2])
    if xy_dist > max_xy_dist:
        return np.inf

    # XY-projected direction angle (sign-agnostic via abs(cos))
    d1 = dir1[:2]
    d2 = dir2[:2]
    n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
    if n1 > 1e-8 and n2 > 1e-8:
        cos_a = np.clip(np.dot(d1, d2) / (n1 * n2), -1.0, 1.0)
        angle = np.degrees(np.arccos(abs(cos_a)))
    else:
        angle = 0.0

    if angle > max_angle_deg:
        return np.inf

    return w_dist * (xy_dist / max_xy_dist) + w_angle * (angle / max_angle_deg)


def match_endpoints(ref_endpoints, mov_endpoints,
                    max_xy_dist=500.0, max_angle_deg=30.0):
    """
    Find an optimal one-to-one matching between two endpoint sets.

    Args:
        ref_endpoints / mov_endpoints: lists returned by
            ``extract_boundary_endpoints``.
        max_xy_dist:   Maximum allowed XY distance (pixels).
        max_angle_deg: Maximum allowed angle between XY directions.

    Returns:
        matches:  list of (ref_idx, mov_idx, cost) tuples.
        ref_xy:   [m, 2]  matched reference XY positions.
        mov_xy:   [m, 2]  matched moving XY positions.
    """
    n_ref, n_mov = len(ref_endpoints), len(mov_endpoints)
    if n_ref == 0 or n_mov == 0:
        return [], np.empty((0, 2)), np.empty((0, 2))

    cost = np.full((n_ref, n_mov), np.inf)
    for i, r in enumerate(ref_endpoints):
        for j, m in enumerate(mov_endpoints):
            cost[i, j] = _pair_score(
                r["pos"], r["dir"], m["pos"], m["dir"],
                max_xy_dist, max_angle_deg,
            )

    finite = cost[np.isfinite(cost)]
    if finite.size == 0:
        return [], np.empty((0, 2)), np.empty((0, 2))

    big = finite.max() * 10 + 100
    cost_filled = np.where(np.isfinite(cost), cost, big)

    row_ind, col_ind = linear_sum_assignment(cost_filled)

    matches, ref_xy, mov_xy = [], [], []
    for r, c in zip(row_ind, col_ind):
        if np.isfinite(cost[r, c]):
            matches.append((r, c, cost[r, c]))
            ref_xy.append(ref_endpoints[r]["pos"][:2])
            mov_xy.append(mov_endpoints[c]["pos"][:2])

    ref_xy = np.array(ref_xy) if ref_xy else np.empty((0, 2))
    mov_xy = np.array(mov_xy) if mov_xy else np.empty((0, 2))
    return matches, ref_xy, mov_xy

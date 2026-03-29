"""
Extract microtubule endpoints near section boundaries with direction vectors.

Based on the projection approach from Weber et al., PLoS ONE 2014
(github.com/zibamira/microtubulestitching).
"""

import numpy as np
from collections import defaultdict


def extract_boundary_endpoints(coords, boundary="bottom", z_band_fraction=0.15,
                               min_direction_pts=3):
    """
    For each MT, find the endpoint closest to the given boundary and compute
    a local direction vector.

    Args:
        coords:            [n, 4] array  —  [id, x, y, z].
        boundary:          'bottom' (high-Z face) or 'top' (low-Z face).
        z_band_fraction:   Fraction of total Z-range that defines the
                           boundary zone.  Only MTs whose endpoint falls
                           inside this zone are returned.
        min_direction_pts: Minimum number of points used for the direction
                           estimate.

    Returns:
        List of dicts, each with keys:
            id  – MT segment ID
            pos – [x, y, z]  endpoint position
            dir – [dx, dy, dz]  unit direction vector toward the boundary
    """
    if coords is None or len(coords) == 0:
        return []

    z_min, z_max = coords[:, 3].min(), coords[:, 3].max()
    z_range = z_max - z_min
    if z_range == 0:
        return []

    z_band = z_range * z_band_fraction
    if boundary == "bottom":
        z_threshold = z_max - z_band
    else:
        z_threshold = z_min + z_band

    # Group points by MT ID
    mt_groups = defaultdict(list)
    for row in coords:
        mt_groups[int(row[0])].append(row)

    endpoints = []
    for mt_id, points in mt_groups.items():
        pts = np.array(points)
        pts = pts[np.argsort(pts[:, 3])]  # sort ascending Z

        if boundary == "bottom":
            endpoint = pts[-1]
            if endpoint[3] < z_threshold:
                continue
            # Last N points closest to boundary
            n = min(len(pts), max(min_direction_pts, int(len(pts) * 0.2)))
            seg = pts[-n:]
        else:
            endpoint = pts[0]
            if endpoint[3] > z_threshold:
                continue
            n = min(len(pts), max(min_direction_pts, int(len(pts) * 0.2)))
            seg = pts[:n]

        # Direction: first → last along the selected segment
        if len(seg) >= 2:
            direction = seg[-1, 1:4] - seg[0, 1:4]
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction /= norm
            else:
                direction = np.array([0.0, 0.0, 1.0 if boundary == "bottom" else -1.0])
        else:
            direction = np.array([0.0, 0.0, 1.0 if boundary == "bottom" else -1.0])

        endpoints.append({"id": mt_id, "pos": endpoint[1:4].copy(), "dir": direction})

    return endpoints

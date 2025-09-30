#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2025                                            #
#######################################################################
from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors

import torch


def pc_median_dist(pc: np.ndarray, avg_over=False, box_size=0.15) -> float:
    """
    Computes the median nearest neighbor distance for a given point cloud.

    This function calculates the median nearest neighbor distance between points
    in a given 2D or 3D point cloud array. Optionally, it can restrict the computation
    to a subset of points that are within a bounding box region centered around the
    median position of the point cloud. The bounding box dimensions can be scaled
    based on a user-defined `box_size`.

    :param pc: A 2D or 3D point cloud array of shape (N, D), where N is the number
        of points and D is the spatial dimensionality (2 or 3).
    :type pc: np.ndarray
    :param avg_over: Flag to indicate whether to compute the distances over a
        subset of the point cloud within a bounding box. Defaults to False.
    :type avg_over: bool, optional
    :param box_size: Fraction of the bounding box size relative to the point cloud
        extents. Only applicable when `avg_over` is True. Defaults to 0.15.
    :type box_size: float, optional
    :return: The mean of the median nearest neighbor distances.
    :rtype: float
    """
    if isinstance(pc, torch.Tensor):
        pc = pc.cpu().detach().numpy()

    if avg_over:
        # Build BB and offset by 10% from the border
        box_dim = pc.shape[1]

        if box_dim in [2, 3]:
            min_x = np.min(pc[:, 0])
            max_x = np.max(pc[:, 0])
            offset_x = (max_x - min_x) * box_size

            min_y = np.min(pc[:, 1])
            max_y = np.max(pc[:, 1])
            offset_y = (max_y - min_y) * box_size
        else:
            offset_x = 0
            offset_y = 0

        if box_dim == 3:
            min_z = np.min(pc[:, 2])
            max_z = np.max(pc[:, 2])
            offset_z = (max_z - min_z) * box_size
        else:
            offset_z = 0

        x = np.median(pc[:, 0])
        y = np.median(pc[:, 1])

        if box_dim == 3:
            z = np.median(pc[:, 2])
        else:
            z = 0

        voxel = point_in_bb(
            pc,
            min_x=x - offset_x,
            max_x=x + offset_x,
            min_y=y - offset_y,
            max_y=y + offset_y,
            min_z=z - offset_z,
            max_z=z + offset_z,
        )
        pc = pc[voxel]

    # build a NearestNeighbors object for efficient nearest neighbor search
    nn = NearestNeighbors(n_neighbors=2, algorithm="kd_tree").fit(pc)

    if pc.shape[0] < 3:
        return 1.0

    distances, _ = nn.kneighbors(pc)
    distances = distances[:, 1]

    return float(np.mean(distances))


def point_in_bb(
    points: np.ndarray,
    min_x: int,
    max_x: int,
    min_y: int,
    max_y: int,
    min_z: Optional[np.float32] = None,
    max_z: Optional[np.float32] = None,
) -> np.ndarray:
    """
    Determines whether points in a given array fall within a specified bounding box.

    The function evaluates if points, provided as an array, lie within the
    boundaries defined by minimum and maximum values for x, y, and optionally z coordinates.
    It enables the filtering of points based on inclusion within a 2D or 3D bounding box.

    :param points: Array representing the coordinates of points, where each row is a point
        and columns correspond to x, y, and optionally z coordinates.
    :type points: numpy.ndarray
    :param min_x: Minimum x-coordinate boundary of the bounding box.
    :type min_x: int
    :param max_x: Maximum x-coordinate boundary of the bounding box.
    :type max_x: int
    :param min_y: Minimum y-coordinate boundary of the bounding box.
    :type min_y: int
    :param max_y: Maximum y-coordinate boundary of the bounding box.
    :type max_y: int
    :param min_z: Optional, minimum z-coordinate boundary of the bounding box.
    :type min_z: numpy.float32, optional
    :param max_z: Optional, maximum z-coordinate boundary of the bounding box.
    :type max_z: numpy.float32, optional
    :return: A boolean array where each element corresponds to the inclusion of a point in the bounding box.
    :return type: numpy.ndarray
    """
    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)

    if points.shape[0] == 3:
        if min_z is not None or max_z is not None:
            bound_z = np.logical_and(points[:, 2] > min_z, points[:, 2] < max_z)
        else:
            bound_z = np.asarray([True for _ in points[:, 2]])

        bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)
    else:
        bb_filter = np.logical_and(bound_x, bound_y)

    return bb_filter

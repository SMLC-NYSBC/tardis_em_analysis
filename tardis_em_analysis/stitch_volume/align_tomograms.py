#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2025                                            #
#######################################################################
import logging
import time
from datetime import datetime
from os import mkdir
from os.path import isdir, split, splitext, join
from typing import Union
import cv2
from sklearn.cluster import KMeans

import numpy as np
import tifffile.tifffile as tiff
from scipy import optimize
from scipy.ndimage import zoom, affine_transform, shift, gaussian_filter
from scipy.stats import entropy

import SimpleITK as sitk
from pystackreg import StackReg
import imreg_dft as ird

from tardis_em_analysis import version
from tardis_em_analysis.stitch_volume.utils import sort_tomogram_files
from tardis_em.utils.export_data import to_am, to_mrc, NumpyToAmira
from tardis_em.utils.load_data import load_image, ImportDataFromAmira
from tardis_em.utils.logo import print_progress_bar, TardisLogo
from tardis_em.utils.normalization import MeanStdNormalize, RescaleNormalize


class AlignTomograms:
    def __init__(self, images_paths: list, coords_paths: list, output_path: str, method='sift'):
        self.path = split(images_paths[0])

        self.images_path = images_paths
        self.coords_path = coords_paths
        self.output_path = output_path
        self.method = method.lower()

        _valid = ['sift', 'warp', 'powell', 'akaze', 'sitk', 'stackreg', 'logpolar', 'imregdft']
        assert method in _valid, f'Method must be one of {_valid}, but got {method}'
        assert len(self.images_path) == len(self.coords_path), \
            (f'Image and coord path must have same length! '
             f'But images {len(self.images_path)} != coors {len(self.coords_path)}')

        if not isdir(self.output_path):
            mkdir(self.output_path, mode=0o777)

        self.eta_predict = "NA"
        self.tardis_progress = None
        self.title = "Fully-automatic alignment of tomograms and spatial graphs"
        self.log_prediction = None
        self.device = "CPU"
        self.down_scale = None

        self.create_headers()
        self.tardis_progress = TardisLogo()
        self.tardis_progress(title=self.title, text_2=f"Device: {self.device}")

        self.course_aligner = VolumeRidgeRegistration(method=self.method,
                                                      down_scale=1)
        self.save_am_coord = NumpyToAmira()

        self.fixed_data, self.moving_data = None, None
        self.accum_angle, self.accum_tx, self.accum_ty = 0., 0., 0.

    def create_headers(self):
        """
        Creates ASCII headers and initializes logging information for a tomogram alignment.
        """
        mapping = {
            "r": "rotation",
            "t": "translation",
            "s": "scaling"
        }

        len_coord = len([x for x in self.coords_path if x is not None])
        loss = "MSE" if self.method=='powell' else "L2"
        self.log_prediction = [
            "###############################################################################",
            "# TARDIS - Transformer And Rapid Dimensionless Instance Segmentation (R)      #",
            f"# tardis_em v{version}                                                           #",
            f"# MIT License * 2021-{datetime.now().year} | Robert Kiewisz & Tristan Bepler                   #",
            "###############################################################################",
            "",
            "---",
            "Course Alignment Setting:",
            "---",
            "",
            "---Directories---",
            f"Input: {self.path}",
            f"Output: {self.output_path}",
            f'Detected: {len(self.images_path)} tomograms and {len_coord} coordinates files.',
            "",
            "---Alignment Parameters---",
            f"Alignment Optimization algorithm: {self.method}",
            f"Alignment Optimization loss function: {loss}",
            f"Alignment with transformations: rotation & translation & scaling",
            f"Down scale: {self.down_scale}",
            "",
            f"Device: {self.device}",
            "",
            "---",
            "Course Alignment Started:",
            "---",
            "",
            "---",
        ]

    def update_progress(self, idx, metric=None):
        if metric is not None:
            a = np.round(metric['Angle'], 2) if 'Angle' in metric else 'NA'
            x = np.round(metric['Tx'], 2) if 'Tx' in metric else 'NA'
            y = np.round(metric['Ty'], 2) if 'Ty' in metric else 'NA'
            sc = np.round(metric['Scale'], 2) if 'Scale' in metric else 'NA'
            so = np.round(metric['Score'], 2) if 'Score' in metric else 'NA'
        else:
            a = x = y = sc = so = 'NA'
        loss = "MSE" if self.method == 'powell' else "L2"

        self.tardis_progress(title=self.title,
                             text_1=f"Found {len(self.images_path) - 1} images to align! [{self.eta_predict} min ETA]",
                             text_2=f"Device: {self.device}",
                             text_3=f"Image {idx + 1}/{len(self.images_path) - 1}:",
                             text_5=f'  Running alignment with: {self.method} model and {loss} loss function...',
                             text_6=f"  Aligned done with: rotation|translation|scaling; {self.down_scale}x Scaling",
                             text_7=f"  Angle: {a}; Tx: {x}; Ty: {y}; Scale: {sc}; Score: {so}",
                             text_9=f"Aligning image {idx+1} to {idx}...",
                             text_10=print_progress_bar(idx+1, len(self.images_path)-1),
                             )

    def save_log(self):
        with open(join(self.output_path, "course_alignment_log.txt"), "w") as f:
            f.write(" \n".join(self.log_prediction))

    def load_tomogram_pairs(self, dir_img_1: str, dir_coord_1: str,
                            dir_img_2: str, dir_coord_2: str):
        if dir_img_1 is not None:
            self.fixed_data = {'Images': np.ndarray, 'Pixel_Size': float,
                               'Coordinates': Union[np.ndarray, None], 'Amira_Transformation': [0, 0, 0],
                               'Ridge_Transform': dict}

        if dir_img_2 is not None:
            self.moving_data = {'Images': np.ndarray, 'Pixel_Size': float,
                                'Coordinates': Union[np.ndarray, None], 'Amira_Transformation': [0, 0, 0],
                                'Ridge_Transform': dict}

        if dir_coord_1 is None:
            self.fixed_data['Coordinates'] = None
            if dir_img_1 is not None:
                img_1, px_1 = load_image(dir_img_1)
            else:
                img_1, px_1 = None, None
        else:
            if dir_coord_1.endswith(".am"):
                am = ImportDataFromAmira(dir_coord_1, dir_img_1)

                img_1, px_1 = am.get_image()
                self.fixed_data['Coordinates'] = am.get_segmented_points()
                self.fixed_data['Amira_Transformation'] = am.transformation
            else:
                self.fixed_data['Coordinates'] = np.genfromtxt(dir_coord_1, delimiter=",", skip_header=1)
                img_1, px_1 = load_image(dir_img_1)
        self.fixed_data['Image'] = img_1
        self.fixed_data['Pixel_Size'] = px_1

        if self.down_scale is None:
            z_dim, y_dim, x_dim = self.fixed_data['Image'].shape
            max_dim = max(z_dim, y_dim, x_dim)
            self.down_scale = 1
            while (max_dim / self.down_scale) > 500:
                 self.down_scale += 1
            self.course_aligner.down_scale = self.down_scale

            self.create_headers()
            self.update_progress(0, None)

        if dir_coord_2 is None:
            self.moving_data['Coordinates'] = None
            if dir_img_2 is not None:
                img_2, px_2 = load_image(dir_img_2)
            else:
                img_2, px_2 = None, None
        else:
            if dir_coord_2.endswith(".am"):
                am = ImportDataFromAmira(dir_coord_2, dir_img_2)

                img_2, px_2 = am.get_image()
                self.moving_data['Coordinates'] = am.get_segmented_points()
                self.moving_data['Amira_Transformation'] = am.transformation
            else:
                self.moving_data['Coordinates'] = np.genfromtxt(dir_coord_2, delimiter=",", skip_header=1)
                img_2, px_2 = load_image(dir_img_2)
        self.moving_data['Image'] = img_2
        self.moving_data['Pixel_Size'] = px_2

    def save_data(self, i: int):
        img_name = splitext(split(self.images_path[i])[-1])[0]
        img_format = splitext(split(self.images_path[i])[-1])[-1]

        new_image_name = img_name + f'_aligned{img_format}'
        new_image_name = join(self.output_path, new_image_name)

        if img_format == ".am":
            to_am(self.moving_data['Image'], self.moving_data['Pixel_Size'], new_image_name, None)
            self.log_prediction.append(f"    - Saved Tomogram Image Data as [{img_format}] file In: {new_image_name}")
        elif img_format in [".mrc", '.rec']:
            to_mrc(self.moving_data['Image'], self.moving_data['Pixel_Size'], new_image_name)
            self.log_prediction.append(f"    - Saved Tomogram Image Data [{img_format}] file In: {new_image_name}")
        elif img_format in [".tif", '.tiff']:
            tiff.imwrite(new_image_name, self.moving_data['Image'])
            self.log_prediction.append(f"    - Saved Tomogram Image Data [{img_format}] file In: {new_image_name}")
        else:
            self.log_prediction.append(f"    - Not Saved Image Data with [{img_format}] unrecognised")

        if self.coords_path[i] is None:
            self.images_path[i] = new_image_name
            return

        if self.moving_data['Coordinates'] is not None:
            img_name = splitext(split(self.images_path[i])[-1])[0]
            coord_name = splitext(split(self.coords_path[i])[-1])[0]
            coord_format = splitext(split(self.coords_path[i])[-1])[-1]

            coord_new_name = coord_name[:len(img_name)] + '_aligned' + coord_name[len(img_name):] + '.am'
            coord_new_name = join(self.output_path, coord_new_name,)

            if coord_format == ".csv":
                np.savetxt(coord_new_name, self.moving_data['Coordinates'], delimiter=",")
                self.log_prediction.append(f"    - Saved Coordinate Data [{coord_format}] file In:     {coord_new_name}")
            elif coord_format == '.am':
                # self.moving_data['Coordinates'][:, 1:] = self.moving_data['Coordinates'][:, 1:] * self.moving_data['Pixel_Size']

                self.save_am_coord.export_amiraV2(coord_new_name, self.moving_data['Coordinates'])
                self.log_prediction.append(f"    - Saved Coordinate Data [{coord_format}] file In:     {coord_new_name}")
            else:
                self.log_prediction.append(f"   -  Not Saved Coordinate Data with [{coord_format}] unrecognised")

        #     self.coords_path[i] = coord_new_name
        # self.images_path[i] = new_image_name

    def align_tomograms(self, i, metric):
        dir_img_1, dir_coord_1 = self.images_path[i], self.coords_path[i]
        dir_img_2, dir_coord_2 = self.images_path[i+1], self.coords_path[i+1]

        self.load_tomogram_pairs(dir_img_1, dir_coord_1, dir_img_2, dir_coord_2)
        self.log_prediction.append(f"# Aligning tomograms {i+1} from {len(self.images_path) - 1}:")
        self.log_prediction = self.log_prediction + ["  - Loaded fixed and moving data for alignment:",
                                                     f"     - Fix Tomogram: {self.images_path[i]}",
                                                     f"     - Fix Coordinate: {self.coords_path[i]}",
                                                     f"         - Pixel Size: {self.fixed_data['Pixel_Size']}",
                                                     f"         - Dim Tomogram: {self.fixed_data['Image'].shape} with {self.fixed_data['Image'].dtype} dtype",
                                                     f"         - Dim Coordinates: {self.fixed_data['Coordinates'].shape}",
                                                     f"     - Moving Tomogram: {self.images_path[i + 1]}",
                                                     f"     - Moving Coordinate: {self.coords_path[i + 1]}",
                                                     f"         - Pixel Size: {self.moving_data['Pixel_Size']}",
                                                     f"         - Dim Tomogram: {self.moving_data['Image'].shape} with {self.moving_data['Image'].dtype} dtype",
                                                     f"         - Dim Coordinates: {self.moving_data['Coordinates'].shape}",
                                                     "",
                                                     ]
        self.save_log()

        # Align tomogram n to tomogram n+1
        metric = self.course_aligner(self.fixed_data['Image'],
                                   self.moving_data['Image'],
                                   self.moving_data['Coordinates'],
                                   return_aligned=False,
                                   transform_fixed=metric)

        self.moving_data['Ridge_Transform'] = metric
        self.log_prediction = self.log_prediction + ["  - Finished aligning moving tomogram:",
                                                     f"     - Angle: {self.moving_data['Ridge_Transform']['Angle']:.2f}",
                                                     f"     - Tx: {self.moving_data['Ridge_Transform']['Tx']:.2f}",
                                                     f"     - Ty: {self.moving_data['Ridge_Transform']['Ty']:.2f}",
                                                     f"     - Scale: {self.moving_data['Ridge_Transform']['Scale']:.2f}",
                                                     f"     - Score: {self.moving_data['Ridge_Transform']['Score']:.2f}",
                                                     f"     - Aligned Tomogram Shape: {self.moving_data['Image'].shape} with {self.moving_data['Image'].dtype} dtype",
                                                     f"     - Aligned Coordinates Shape: {self.moving_data['Coordinates'].shape}",
                                                     "",
                                                     ]
        self.save_log()
        self.update_progress(i, metric)

        moving_vol = self.moving_data['Image'].shape
        # self.moving_data['Image'] = self.course_aligner.get_ridge_transform(self.moving_data['Image'])
        # self.moving_data['Coordinates'] = self.course_aligner.get_ridge_transform_coord(moving_vol,
        #                                                                                 self.moving_data['Coordinates'],
        #                                                                                 *self.moving_data['Image'].shape[1:])

        # Save tomogram n+1 under the same file format
        self.log_prediction.append("  - Saved aligned moving data:")
        self.save_log()
        # self.save_data(i+1)

        self.log_prediction = self.log_prediction + ["", "---", "",]
        self.save_log()
        return metric

    def align_all_tomograms(self):
        global_start = time.time()

        metric = None
        self.update_progress(0, metric)
        for idx in range(len(self.images_path) - 1):
            start = time.time()
            metric = self.align_tomograms(idx, metric)
            end = time.time()

            self.eta_predict = round(((end - start) * (len(self.images_path) - idx - 1)) / 60, 1)
            self.update_progress(idx, metric)

        dir_img_2, dir_coord_2 = self.images_path[0], self.coords_path[0]

        self.load_tomogram_pairs(None, None, dir_img_2, dir_coord_2)
        self.save_data(0)
        global_end = time.time()
        self.log_prediction = self.log_prediction + [
            '',
            "---",
            f"Total time for aligning all tomograms: {(global_end - global_start) / 60:.2f} minutes",
        ]

    def stitch_align_volumes(self):
        output_path_images, output_path_coords = sort_tomogram_files(self.output_path)

        stitched_ = []
        for i in output_path_images:
            if not i.endswith(("stitched_volume.am", "stitched_volume.mrc", "stitched_volume.rec", "stitched_volume.tif", "stitched_volume.tiff")):
                vol, px = load_image(i, False, True)
                stitched_.append(vol)
        stitched_ = np.concatenate(stitched_, axis=0)
        to_am(stitched_, px, join(self.output_path, 'stitched_volume.am'))

        self.log_prediction = self.log_prediction + ["",
                                                      "---",
                                                     "Stitched final course alignment:",
                                                     "---",
                                                     f"     - Final Tomogram Shape: {stitched_.shape}",
                                                     "",
                                                     ]
        self.save_log()

        stitched_ = []
        last_max_z = 0
        last_max_id = 0
        for i in output_path_coords:
            if i is None:
                continue

            coord = ImportDataFromAmira(i).get_segmented_points()
            coord[:, 0] += last_max_id
            coord[:, -1] += last_max_z
            stitched_.append(coord)

            last_max_id = coord[:, 0].max() + 1
            last_max_z = coord[:, -1].max() + 1

        stitched_ = np.concatenate(stitched_)
        self.save_am_coord.export_amiraV2(join(self.output_path, 'stitched_coord.am'), stitched_)
        self.log_prediction = self.log_prediction + [
                                                     f"     - Final Coordinate Shape: {stitched_.shape}",
                                                     f"     - Final MT number: {np.max(stitched_[:, 0])}",
                                                     ]
        self.save_log()


class VolumeRidgeRegistration:
    def __init__(
            self,
            method='sift',
            down_scale=6,
            log_=False,
    ):
        method = method.lower()
        _valid = ['sift', 'warp', 'powell', 'akaze', 'sitk', 'stackreg', 'logpolar', 'imregdft']
        assert method in _valid, f'Method must be one of {_valid}, but got {method}'
        self.mean_std = MeanStdNormalize()
        self.normalize = RescaleNormalize(clip_range=(.1, 99.9))

        self.method = method
        # Mutual information is more robust for EM cross-section alignment
        # where tissue may be partially lost between sections
        self.optimize_fn = 'mi' if method == 'powell' else 'mse'

        self.down_scale = down_scale
        self.ridge_operation = 'rst'

        self.Angle, self.Ty, self.Tx, self.Scale, self.Score = 0.0, 0.0, 0.0, 1.0, 0.0

        self.mask_fix, self.mask_moving = None, None
        self.img2_y, self.img2_x = 0, 0

        self.log_ = log_

    def volume_to_projection(self, img1, img2, original_=False, transform_fixed=None):
        # Adaptive slice count: 5% of depth, clamped to [5, 5]
        n_slices_1 = max(5, min(5, int(img1.shape[0] * 0.05)))
        n_slices_2 = max(5, min(5, int(img2.shape[0] * 0.05)))

        # Mean projection: more stable than sum, avoids overflow, better SNR
        img1 = np.sum(img1[-n_slices_1:, ...], axis=0).astype(np.float32)
        img1 = zoom(img1, 1 / self.down_scale)
        img2 = np.sum(img2[:n_slices_2, ...], axis=0).astype(np.float32)
        img2 = zoom(img2, 1 / self.down_scale)

        # Store dimensions after downscale (used for Powell bounds)
        self.img2_y, self.img2_x = img2.shape

        if not original_:
            img1, img2 = gaussian_filter(img1, sigma=1.5), gaussian_filter(img2, sigma=1.5)
            img1 = self.normalize((self.mean_std(img1)).astype(np.float32))
            img1 = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-10)
            img1 = np.clip(img1, 0, 1)
            img2 = self.normalize((self.mean_std(img2)).astype(np.float32))
            img2 = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-10)
            img2 = np.clip(img2, 0, 1)

            self.mask_fix = np.ones_like(img1, dtype=np.uint8)
            self.mask_moving = np.ones_like(img2, dtype=np.uint8)

        pad = int((np.sqrt(img1.shape[0] ** 2 + img1.shape[1] ** 2)) - img1.shape[0]) // 2
        img1 = np.pad(img1, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
        img2 = np.pad(img2, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)

        if not original_:
            self.mask_fix = np.pad(self.mask_fix, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
            self.mask_moving = np.pad(self.mask_moving, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)

        if transform_fixed is not None:
            img1 = self.apply_rigid_transform(img1,
                                              transform_fixed['Angle'],
                                              transform_fixed['Tx'] / self.down_scale,
                                              transform_fixed['Ty'] / self.down_scale,
                                              transform_fixed['Scale'])

            if not original_:
                self.mask_fix = self.apply_rigid_transform(self.mask_fix,
                                                  transform_fixed['Angle'],
                                                  transform_fixed['Tx'] / self.down_scale,
                                                  transform_fixed['Ty'] / self.down_scale,
                                                  transform_fixed['Scale'])
                self.mask_fix = np.where(self.mask_fix > 0, 1, 0).astype(np.uint8)

        return img1, img2

    @staticmethod
    def compute_padding(h, w, angle, tx, ty, scale, _2d=True):
        # 1. Scale
        w_s, h_s = w * scale, h * scale

        # 2. Rotate
        theta = np.deg2rad(angle)
        w_r = abs(w_s * np.cos(theta)) + abs(h_s * np.sin(theta))
        h_r = abs(w_s * np.sin(theta)) + abs(h_s * np.cos(theta))

        # 3. Translate
        pad_left = (w_r - w) / 2 + max(0, tx)
        pad_right = (w_r - w) / 2 + max(0, -tx)
        pad_H = int(max(pad_left, pad_right))

        pad_top = (h_r - h) / 2 + max(0, ty)
        pad_bottom = (h_r - h) / 2 + max(0, -ty)
        pad_W = int(max(pad_top, pad_bottom))

        if _2d:
            return (pad_H, pad_H), (pad_W, pad_W)
        return (0, 0), (pad_H, pad_H), (pad_W, pad_W)

    @staticmethod
    def translate_point_cloud(points, angle_deg, distance, inv_=False):
        theta = np.deg2rad(angle_deg)
        dx = distance * np.cos(theta)
        dy = distance * np.sin(theta)

        points_copy = points.copy()
        if inv_:
            points_copy[:, 1] += dy  # X
            points_copy[:, 2] += dx  # Y
        else:
            points_copy[:, 1] += dx  # X
            points_copy[:, 2] += dy  # Y

        return points_copy

    def loss_function_ssim(self, img1, img2, valid_mask_2d, sigma=1.5, C1=0.01 ** 2, C2=0.03 ** 2):
        """Compute SSIM loss on full 2D images, averaged over valid mask region."""
        img1 = img1.astype(float)
        img2 = img2.astype(float)

        mu1 = gaussian_filter(img1, sigma)
        mu2 = gaussian_filter(img2, sigma)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = gaussian_filter(img1 ** 2, sigma) - mu1_sq
        sigma2_sq = gaussian_filter(img2 ** 2, sigma) - mu2_sq
        sigma12 = gaussian_filter(img1 * img2, sigma) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                    (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if valid_mask_2d.sum() == 0:
            return 0.0

        loss = -np.mean(ssim_map[valid_mask_2d])

        if self.log_:
            print(loss)
        return loss

    def loss_function_l2(self, vol1, vol2, valid_mask):
        vol1 = vol1.astype(float)[valid_mask]
        vol2 = vol2.astype(float)[valid_mask]

        loss = np.linalg.norm(vol1 - vol2)

        if self.log_:
            print(loss)
        return loss

    def loss_function_mean_squared_error(self, img1, img2, valid_mask):
        """Compute Mean Squared Error between two images."""
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions for MSE")
        img1 = img1.astype(float)[valid_mask]
        img2 = img2.astype(float)[valid_mask]

        loss = np.mean((img1 - img2) ** 2)

        if self.log_:
            print(loss)

        return loss

    def loss_function_mutual_information(self, img1, img2, valid_mask, bins=256):
        """Compute Mutual Information between two images."""
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions for MI")

        # Normalize images to [0, 1] for histogra
        img1 = img1[valid_mask]
        img2 = img2[valid_mask]

        img1 = 255 * (img1 - img1.min()) / (img1.max() - img1.min())
        img2 = 255 * (img2 - img2.min()) / (img2.max() - img2.min())

        # Compute joint histogram
        hist, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins)
        joint_prob = hist / np.sum(hist)

        # Compute entropies with masking for zero probabilities
        p1 = np.sum(joint_prob, axis=1)
        p2 = np.sum(joint_prob, axis=0)
        h1 = entropy(p1)
        h2 = entropy(p2)
        h_joint = entropy(joint_prob.ravel())

        # Mutual Information
        loss = -(h1 + h2 - h_joint)

        if self.log_:
            print(loss)

        return loss

    def loss_function_normalized_cross_correlation(self, img1, img2, valid_mask):
        """Compute Normalized Cross-Correlation between two images over valid region."""
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions for NCC")

        v1 = img1[valid_mask].astype(float)
        v2 = img2[valid_mask].astype(float)

        v1 = (v1 - np.mean(v1)) / (np.std(v1) + 1e-10)
        v2 = (v2 - np.mean(v2)) / (np.std(v2) + 1e-10)

        loss = -np.mean(v1 * v2)

        if self.log_:
            print(loss)

        return loss

    @staticmethod
    def compute_transformation_matrix(cy, cx, angle, scale):
        # Convert angle to radians
        theta = np.deg2rad(angle)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        # Affine matrix for rotation + translation
        cy, cx = cy / 2, cx / 2

        # Translate image center to origin
        T1 = np.array([[1, 0, -cx],
                       [0, 1, -cy],
                       [0, 0, 1]])

        # Rotation matrix
        RS = np.array([[scale * cos_theta, -scale * sin_theta, 0],
                      [scale * sin_theta, scale * cos_theta, 0],
                      [0, 0, 1]])

        # Translate back and apply user translation
        T2 = np.array([[1, 0, cx],
                       [0, 1, cy],
                       [0, 0, 1]])

        # Combined matrix: move -> rotate -> scale -> move back + translate
        return T2 @ RS @ T1

    def apply_rigid_transform(self, img, angle, tx, ty, scale, cval=False, reshape=False):
        """Apply rotation and translation to an image without cropping."""
        if cval:
            min_ = img.min()
            if min_ > 0:
                min_ = 0
        else:
            min_ = 0

        if img.ndim == 2:
            if reshape:
                pad = self.compute_padding(img.shape[0], img.shape[1], angle, 0., 0., scale, _2d=True)
                img = np.pad(img, pad, 'constant', constant_values=cval)

            M = self.compute_transformation_matrix(*img.shape, angle, scale)
            M = np.linalg.inv(M)

            img = affine_transform(
                img,
                M[:2, :2],  # 2x2 rotation matrix
                offset=M[:2, 2],  # Combine offset and translation
                output_shape=img.shape,
                order=3,
                mode="constant",
                cval=min_,
            )

            if reshape:
                y_max, x_max = int(np.abs(ty)), int(np.abs(tx))
                img = np.pad(img, ((y_max, y_max), (x_max, x_max)), 'constant', constant_values=cval)
            img = shift(img,
                        (ty, tx),
                        order=3,
                        mode="constant",
                        cval=min_,
                        )
        else:
            if reshape:
                pad = self.compute_padding(img.shape[1], img.shape[2], angle, 0., 0., scale, _2d=False)
                img = np.pad(img, pad, 'constant', constant_values=cval)

            M = self.compute_transformation_matrix(*img.shape[1:], angle, scale)
            M = np.linalg.inv(M)

            for i in range(img.shape[0]):
                img[i] = affine_transform(
                    img[i],
                    M[:2, :2],  # 2x2 rotation matrix
                    offset=M[:2, 2],  # Combine offset and translation
                    output_shape=img.shape[1:],
                    order=3,
                    mode="constant",
                    cval=min_,
                )

            if reshape:
                y_max, x_max = int(np.abs(ty)), int(np.abs(tx))
                img = np.pad(img, ((0, 0), (y_max, y_max), (x_max, x_max)), 'constant', constant_values=cval)

            for i in range(img.shape[0]):
                img[i] = shift(img[i],
                               (ty, tx),
                               order=3,
                               mode="constant",
                               cval=min_,
                               )
        return img

    def find_best_ridge_transformation(self, params, fixed, moving):
        """Objective function to minimize for intensity-based registration."""
        moving_T = self.apply_rigid_transform(moving, *params)

        # Zero borders to avoid interpolation artefacts
        moving_T[:2, :] = 0
        moving_T[-2:, :] = 0
        moving_T[:, :2] = 0
        moving_T[:, -2:] = 0

        # Build valid region mask (2D boolean)
        moved_mask = self.apply_rigid_transform(self.mask_moving, *params)
        valid_2d = np.logical_and(moved_mask > 0, self.mask_fix > 0)
        valid_idx = np.where(valid_2d)

        if valid_idx[0].size == 0:
            return 1e10

        # Normalize both images symmetrically within the valid region
        f_mean, f_std = np.mean(fixed[valid_idx]), np.std(fixed[valid_idx]) + 1e-10
        m_mean, m_std = np.mean(moving_T[valid_idx]), np.std(moving_T[valid_idx]) + 1e-10

        fixed_norm = fixed.copy()
        fixed_norm[valid_idx] = (fixed_norm[valid_idx] - f_mean) / f_std
        moving_T[valid_idx] = (moving_T[valid_idx] - m_mean) / m_std

        if self.optimize_fn == "mi":
            # MI is invariant to intensity transforms, use unnormalized
            return self.loss_function_mutual_information(fixed, moving_T, valid_idx)
        elif self.optimize_fn == "mse":
            return self.loss_function_mean_squared_error(fixed_norm, moving_T, valid_idx)
        elif self.optimize_fn == "l2":
            return self.loss_function_l2(fixed_norm, moving_T, valid_idx)
        elif self.optimize_fn == "ncc":
            return self.loss_function_normalized_cross_correlation(fixed_norm, moving_T, valid_idx)
        elif self.optimize_fn == "ssim":
            # SSIM needs full 2D images with boolean mask
            return self.loss_function_ssim(fixed_norm, moving_T, valid_2d)
        elif self.optimize_fn == "mse_mi":
            return (self.loss_function_mean_squared_error(fixed_norm, moving_T, valid_idx)
                    + self.loss_function_mutual_information(fixed, moving_T, valid_idx))

    def optim_align_images(self, fixed_img, moving_img):
        """Align moving_img to fixed_img using feature-based or intensity-based registration."""
        if self.method in ["sift", "akaze"]:
            # Convert to uint8 with CLAHE for better feature detection in low-SNR EM
            fixed_uint8 = np.clip(fixed_img * 255.0, 0, 255).astype(np.uint8)
            moving_uint8 = np.clip(moving_img * 255.0, 0, 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            fixed_uint8 = clahe.apply(fixed_uint8)
            moving_uint8 = clahe.apply(moving_uint8)

            # OpenCV masks must be 0/255 uint8
            mask_fix = (np.clip(self.mask_fix, 0, 1) * 255).astype(np.uint8)
            mask_moving = (np.clip(self.mask_moving, 0, 1) * 255).astype(np.uint8)

            if self.method == "sift":
                self.Angle, self.Tx, self.Ty, self.Scale = self.align_images_sift(
                    fixed_uint8, moving_uint8, mask_fix, mask_moving)
            else:
                self.Angle, self.Tx, self.Ty, self.Scale = self.align_images_akaze(
                    fixed_uint8, moving_uint8, mask_fix, mask_moving)

        elif self.method == "warp":
            mask_moving_uint8 = (np.clip(self.mask_moving, 0, 1) * 255).astype(np.uint8)
            self.Angle, self.Tx, self.Ty, self.Scale = self.align_images_warp(
                fixed_img, moving_img, mask_moving_uint8)

        elif self.method == 'powell':
            # Use SIFT as initialization to avoid local minima
            try:
                fixed_uint8 = np.clip(fixed_img * 255.0, 0, 255).astype(np.uint8)
                moving_uint8 = np.clip(moving_img * 255.0, 0, 255).astype(np.uint8)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                fixed_clahe = clahe.apply(fixed_uint8)
                moving_clahe = clahe.apply(moving_uint8)
                mask_fix = (np.clip(self.mask_fix, 0, 1) * 255).astype(np.uint8)
                mask_moving = (np.clip(self.mask_moving, 0, 1) * 255).astype(np.uint8)
                init_a, init_tx, init_ty, init_s = self.align_images_sift(
                    fixed_clahe, moving_clahe, mask_fix, mask_moving)
                # Fall back to identity if SIFT returned degenerate result
                if init_s < 0.5 or init_s > 2.0:
                    init_a, init_tx, init_ty, init_s = 0.0, 0.0, 0.0, 1.0
            except Exception:
                init_a, init_tx, init_ty, init_s = 0.0, 0.0, 0.0, 1.0

            initial_params = [init_a, init_tx, init_ty, init_s]

            bounds = [
                (-180.0, 180.0) if "r" in self.ridge_operation else (0.0, 0.0),
                (
                    (-self.img2_x // 2, self.img2_x // 2)
                    if "t" in self.ridge_operation
                    else (0.0, 0.0)
                ),
                (
                    (-self.img2_y // 2, self.img2_y // 2)
                    if "t" in self.ridge_operation
                    else (0.0, 0.0)
                ),
                (0.85, 1.15) if "s" in self.ridge_operation else (1.0, 1.0),
            ]

            result = optimize.minimize(
                self.find_best_ridge_transformation,
                initial_params,
                args=(fixed_img, moving_img),
                method='powell',
                bounds=bounds,
                tol=1e-9,
            )

            self.Angle, self.Tx, self.Ty, self.Scale = result.x
            self.Score = result.fun

        elif self.method == 'sitk':
            self.Angle, self.Tx, self.Ty, self.Scale = self.align_images_sitk(
                fixed_img, moving_img)

        elif self.method == 'stackreg':
            self.Angle, self.Tx, self.Ty, self.Scale = self.align_images_stackreg(
                fixed_img, moving_img)

        elif self.method == 'logpolar':
            self.Angle, self.Tx, self.Ty, self.Scale = self.align_images_logpolar(
                fixed_img, moving_img)

        elif self.method == 'imregdft':
            self.Angle, self.Tx, self.Ty, self.Scale = self.align_images_imregdft(
                fixed_img, moving_img)

        # Scale translations from downscaled projection space to full-resolution
        self.Ty *= self.down_scale
        self.Tx *= self.down_scale

    def align_images_sift(self, reference: np.ndarray, moving: np.ndarray,
                          mask_fix: np.ndarray = None, mask_moving: np.ndarray = None,
                          max_features=500, good_match_ratio=0.75):
        # Detect SIFT features and descriptors
        sift = cv2.SIFT_create()
        kp_ref, des_ref = sift.detectAndCompute(reference, mask_fix)
        kp_mov, des_mov = sift.detectAndCompute(moving, mask_moving)

        if des_ref is None or des_mov is None or len(des_ref) < 2 or len(des_mov) < 2:
            return 0, 0, 0, 1

        # Match features using BFMatcher with a ratio test
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        raw_matches = matcher.knnMatch(des_mov, des_ref, k=2)  # Query: moving, Train: reference

        # Apply Lowe's ratio test for good matches
        good_matches = []
        for m, n in raw_matches:
            if m.distance < good_match_ratio * n.distance:
                good_matches.append(m)

        # Sort by distance and limit if needed
        good_matches = sorted(good_matches, key=lambda x: x.distance)[:max_features]

        if len(good_matches) < 4:  # Need at least 4 for similarity
            return 0, 0, 0, 1

        # Extract points
        pts_mov = np.float32([kp_mov[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimate similarity transformation (rotation, translation, scale) with RANSAC
        matrix, inliers = cv2.estimateAffinePartial2D(pts_mov, pts_ref, method=cv2.RANSAC)

        if matrix is None:
            return 0, 0, 0, 1

        scale = np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2)

        # Reject degenerate results (scale near 0 or wildly off)
        if scale < 0.5 or scale > 2.0:
            return 0, 0, 0, 1

        angle = np.arctan2(matrix[0, 1], matrix[0, 0]) * 180 / np.pi
        tx, ty = matrix[0, 2], matrix[1, 2]

        return angle, tx, ty, scale

    def align_images_akaze(self, reference: np.ndarray, moving: np.ndarray,
                           mask_fix: np.ndarray = None, mask_moving: np.ndarray = None,
                           max_features=500, good_match_ratio=0.75):
        # Detect AKAZE features and descriptors
        akaze = cv2.AKAZE_create()
        kp_ref, des_ref = akaze.detectAndCompute(reference, mask_fix)
        kp_mov, des_mov = akaze.detectAndCompute(moving, mask_moving)

        if des_ref is None or des_mov is None or len(des_ref) < 2 or len(des_mov) < 2:
            return 0, 0, 0, 1

        # Match features using BFMatcher with Hamming distance
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw_matches = matcher.knnMatch(des_mov, des_ref, k=2)  # Query: moving, Train: reference

        # Apply ratio test for good matches
        good_matches = []
        for m, n in raw_matches:
            if m.distance < good_match_ratio * n.distance:
                good_matches.append(m)

        # Sort by distance and limit if needed
        good_matches = sorted(good_matches, key=lambda x: x.distance)[:max_features]

        if len(good_matches) < 4:  # Need at least 4 for similarity
            return 0, 0, 0, 1

        # Extract points
        pts_mov = np.float32([kp_mov[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimate similarity transformation (rotation, translation, scale) with RANSAC
        matrix, inliers = cv2.estimateAffinePartial2D(pts_mov, pts_ref, method=cv2.RANSAC)

        if matrix is None:
            return 0, 0, 0, 1

        scale = np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2)

        # Reject degenerate results (scale near 0 or wildly off)
        if scale < 0.5 or scale > 2.0:
            return 0, 0, 0, 1

        angle = np.arctan2(matrix[0, 1], matrix[0, 0]) * 180 / np.pi
        tx, ty = matrix[0, 2], matrix[1, 2]

        return angle, tx, ty, scale

    def align_images_warp(self, reference: np.ndarray, moving: np.ndarray,
                          mask_moving: np.ndarray = None, max_iter=1000, epsilon=1e-4):
        # warp_mode = cv2.MOTION_HOMOGRAPHY
        # warp_matrix = np.eye(3, dtype=np.float32)
        warp_mode = cv2.MOTION_EUCLIDEAN     # or even MOTION_EUCLIDEAN for start
        warp_matrix = np.eye(2, 3, dtype=np.float32)   # 2×3

        
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, epsilon)

        def normalize_for_ecc(img):
            img = img.astype(np.float32)
            img -= np.mean(img)
            img /= (np.std(img) + 1e-8)          # avoid div-by-zero
            return img

        reference_norm = normalize_for_ecc(reference)
        moving_norm    = normalize_for_ecc(moving)

        _, warp_matrix = cv2.findTransformECC(reference_norm, moving_norm, warp_matrix, warp_mode, criteria, mask_moving)

        if np.isnan(warp_matrix).any():
            return 0.0, 0.0, 0.0, 1.0

        angle = np.arctan2(warp_matrix[0, 1], warp_matrix[0, 0]) * 180 / np.pi
        tx, ty = warp_matrix[0, 2], warp_matrix[1, 2]
        scale = np.sqrt(warp_matrix[0, 0] ** 2 + warp_matrix[0, 1] ** 2)

        return angle, tx, ty, scale

    def _crop_to_content(self, fixed_img, moving_img):
        """Crop both images to the bounding box of their combined content masks.

        The projections are zero-padded for rotation tolerance. This method
        extracts the content region so registration methods aren't dominated
        by the identical zero borders.

        Returns (fixed_crop, moving_crop, offset_y, offset_x).
        """
        # Union of both masks to get the full content region
        combined = np.maximum(self.mask_fix, self.mask_moving)
        rows = np.any(combined, axis=1)
        cols = np.any(combined, axis=0)

        if not rows.any() or not cols.any():
            # Fallback: return originals if masks are empty
            return fixed_img, moving_img, 0, 0

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        fixed_crop = fixed_img[rmin:rmax + 1, cmin:cmax + 1]
        moving_crop = moving_img[rmin:rmax + 1, cmin:cmax + 1]

        return fixed_crop, moving_crop, rmin, cmin

    @staticmethod
    def _apply_clahe(img):
        """Apply CLAHE to a [0,1] float image. Returns float64 in [0,1]."""
        uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(uint8).astype(np.float64) / 255.0

    def align_images_sitk(self, fixed_img, moving_img):
        """SimpleITK Similarity2D registration with Mattes MI + multi-resolution."""
        # Crop to content region to avoid zero-padded borders dominating registration
        fixed_crop, moving_crop, offset_y, offset_x = self._crop_to_content(fixed_img, moving_img)

        # CLAHE enhances local contrast for better MI gradient signal
        fixed_f64 = self._apply_clahe(fixed_crop)
        moving_f64 = self._apply_clahe(moving_crop)

        fixed_sitk = sitk.GetImageFromArray(fixed_f64)
        moving_sitk = sitk.GetImageFromArray(moving_f64)
        fixed_sitk = sitk.Cast(fixed_sitk, sitk.sitkFloat64)
        moving_sitk = sitk.Cast(moving_sitk, sitk.sitkFloat64)

        # MOMENTS aligns centers-of-mass and principal axes → better start than GEOMETRY
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_sitk, moving_sitk,
            sitk.Similarity2DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS
        )

        reg = sitk.ImageRegistrationMethod()

        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        reg.SetMetricSamplingStrategy(reg.REGULAR)
        reg.SetMetricSamplingPercentage(0.5)

        reg.SetInterpolator(sitk.sitkLinear)

        # Deeper pyramid: start very coarse to avoid local minima
        reg.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 2, 1])
        reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[4.0, 2.0, 1.0, 0.0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # GradientDescentLineSearch: line search finds optimal step automatically,
        # much more robust than RegularStepGradientDescent which halves step on
        # direction change and can stall when initial gradient is small.
        reg.SetOptimizerAsGradientDescentLineSearch(
            learningRate=1.5,
            numberOfIterations=500,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=20,
        )
        reg.SetOptimizerScalesFromPhysicalShift()

        reg.SetInitialTransform(initial_transform, inPlace=False)

        try:
            final_transform = reg.Execute(fixed_sitk, moving_sitk)
        except RuntimeError:
            return 0.0, 0.0, 0.0, 1.0

        # Extract parameters from Similarity2DTransform
        if isinstance(final_transform, sitk.CompositeTransform):
            t = sitk.Similarity2DTransform(final_transform.GetNthTransform(0))
        else:
            t = sitk.Similarity2DTransform(final_transform)

        angle = np.rad2deg(t.GetAngle())
        scale = t.GetScale()
        # SimpleITK returns (tx, ty) in physical coords (col, row)
        tx, ty = t.GetTranslation()

        self.Score = reg.GetMetricValue()

        return angle, tx, ty, scale

    def align_images_stackreg(self, fixed_img, moving_img):
        """pyStackReg SCALED_ROTATION: proven for microscopy stack alignment."""
        # Crop to content region to avoid zero-padded borders dominating
        fixed_crop, moving_crop, offset_y, offset_x = self._crop_to_content(fixed_img, moving_img)

        # CLAHE enhances local contrast → better feature detection in TurboReg
        fixed_enh = self._apply_clahe(fixed_crop)
        moving_enh = self._apply_clahe(moving_crop)

        sr = StackReg(StackReg.SCALED_ROTATION)
        tmat = sr.register(fixed_enh, moving_enh)

        # Extract angle, scale, tx, ty from the 3x3 similarity matrix
        scale = np.sqrt(tmat[0, 0] ** 2 + tmat[1, 0] ** 2)
        angle = np.arctan2(tmat[1, 0], tmat[0, 0]) * 180.0 / np.pi
        tx = tmat[0, 2]
        ty = tmat[1, 2]

        if scale < 0.5 or scale > 2.0:
            return 0.0, 0.0, 0.0, 1.0

        return angle, tx, ty, scale

    def align_images_logpolar(self, fixed_img, moving_img):
        """Log-polar + phase correlation: FFT-based rotation+scale+translation recovery."""
        from skimage.registration import phase_cross_correlation
        from skimage.transform import warp_polar

        # Crop to content region to avoid zero-padded borders
        fixed_f, moving_f, offset_y, offset_x = self._crop_to_content(fixed_img, moving_img)
        fixed_f = fixed_f.astype(np.float64)
        moving_f = moving_f.astype(np.float64)

        # Apply Hanning window to reduce spectral leakage before FFT
        h, w = fixed_f.shape
        window = np.outer(np.hanning(h), np.hanning(w))

        # Step 1: Use magnitude spectrum to decouple translation from rotation/scale
        f_fixed = np.fft.fftshift(np.abs(np.fft.fft2(fixed_f * window)))
        f_moving = np.fft.fftshift(np.abs(np.fft.fft2(moving_f * window)))

        # Apply log on magnitude (improves dynamic range)
        f_fixed = np.log1p(f_fixed)
        f_moving = np.log1p(f_moving)

        # Step 2: Log-polar transform of magnitude spectra
        radius = min(f_fixed.shape) // 2
        lp_fixed = warp_polar(f_fixed, radius=radius, scaling='log', output_shape=(360, radius))
        lp_moving = warp_polar(f_moving, radius=radius, scaling='log', output_shape=(360, radius))

        # Step 3: Phase correlation on log-polar images → recovers angle and log(scale)
        shift_lp, _, _ = phase_cross_correlation(
            lp_fixed, lp_moving, upsample_factor=20, normalization=None
        )

        angle = -shift_lp[0]  # row = angle (1 row = 1 degree)
        # Convert log-scale shift to actual scale factor
        log_base = np.log(radius) / radius
        scale = np.exp(shift_lp[1] * log_base)

        if scale < 0.5 or scale > 2.0:
            scale = 1.0

        # Step 4: De-rotate and de-scale the moving image, then find translation
        from skimage.transform import rotate as sk_rotate, rescale as sk_rescale
        moving_corrected = sk_rotate(moving_f, angle, resize=False, preserve_range=True)
        if abs(scale - 1.0) > 0.01:
            moving_corrected = zoom(moving_corrected, 1.0 / scale)
            # Pad or crop to match fixed size
            dy = fixed_f.shape[0] - moving_corrected.shape[0]
            dx = fixed_f.shape[1] - moving_corrected.shape[1]
            if dy > 0 or dx > 0:
                moving_corrected = np.pad(moving_corrected,
                    ((max(0, dy // 2), max(0, dy - dy // 2)),
                     (max(0, dx // 2), max(0, dx - dx // 2))),
                    mode='constant')
            moving_corrected = moving_corrected[:fixed_f.shape[0], :fixed_f.shape[1]]

        # Apply Hanning window for translation phase correlation too
        h2, w2 = fixed_f.shape
        window2 = np.outer(np.hanning(h2), np.hanning(w2))
        shift_trans, _, _ = phase_cross_correlation(
            fixed_f * window2, moving_corrected * window2,
            upsample_factor=20, normalization=None
        )
        ty, tx = shift_trans[0], shift_trans[1]

        return angle, tx, ty, scale

    def align_images_imregdft(self, fixed_img, moving_img):
        """imreg_dft: FFT-based similarity registration (Reddy & Chatterji 1996)."""
        # Crop to content region to avoid zero-padded borders
        fixed_crop, moving_crop, offset_y, offset_x = self._crop_to_content(fixed_img, moving_img)

        # CLAHE enhances contrast for better FFT peak detection
        fixed_f = self._apply_clahe(fixed_crop)
        moving_f = self._apply_clahe(moving_crop)

        try:
            result = ird.similarity(fixed_f, moving_f, numiter=5)
        except Exception:
            return 0.0, 0.0, 0.0, 1.0

        angle = result['angle']
        scale = result['scale']
        ty, tx = result['tvec']  # (row, col) = (ty, tx)
        self.Score = result.get('success', 0.0)

        if scale < 0.5 or scale > 2.0:
            return 0.0, 0.0, 0.0, 1.0

        return angle, float(tx), float(ty), scale

    def get_transformation_metrics(self):
        return dict(
            zip(
                ["Angle", "Tx", "Ty", "Scale", "Score"],
                [self.Angle, self.Tx, self.Ty, self.Scale, self.Score],
            )
        )

    def get_ridge_transform(self, moving_vol, reshape=False):
        return self.apply_rigid_transform(moving_vol, self.Angle, self.Tx, self.Ty, self.Scale, cval=True, reshape=reshape)

    def get_ridge_transform_coord(self, moving_vol_shape, moving_coord, cy, cx):
        if moving_vol_shape[-2] != cy and moving_vol_shape[-1] != cx:
            adjust_padding = True
        else:
            adjust_padding = False

        if adjust_padding:
            pad_H, pad_W = self.compute_padding(*moving_vol_shape[1:], self.Angle, 0., 0., self.Scale)

            moving_coord[:, 1] = moving_coord[:, 1] + pad_H[0]
            moving_coord[:, 2] = moving_coord[:, 2] + pad_W[0]

        M = self.compute_transformation_matrix(cy, cx, self.Angle, self.Scale)
        M = np.linalg.inv(M)

        coords = np.ones((moving_coord.shape[0], 3))
        coords[:, 0] = np.copy(moving_coord[:, 1])
        coords[:, 1] = np.copy(moving_coord[:, 2])

        # Apply transformation
        coords = (M @ coords.T).T[:, :2]
        coords = np.column_stack((
            moving_coord[:, 0],  # label
            ((coords[:, 0] - (cx/2)) * self.Scale + (cx/2)),  # + self.tx,  # new_x
            ((coords[:, 1] - (cy/2)) * self.Scale + (cy/2)),  # + self.ty,  # new_y
            moving_coord[:, -1]  # z (unchanged)
        ))

        if 't' in self.ridge_operation and adjust_padding:
            y_max, x_max = np.abs(self.Ty) + 3, np.abs(self.Tx) + 3
            coords = self.translate_point_cloud(coords, -self.Angle, x_max)
            coords = self.translate_point_cloud(coords, self.Angle, y_max, inv_=True)

        coords[:, 1] += self.Tx
        coords[:, 2] += self.Ty

        return coords

    def update_log(self, log_: dict):
        for k, v in log_.items():
            setattr(self, k, v)

    def __call__(self, fixed_vol, moving_vol, moving_coord=None, return_aligned=True, transform_fixed=None):
        self.Angle, self.Ty, self.Tx, self.Scale, self.Score = 0.0, 0.0, 0.0, 1.0, 0.0

        self.optim_align_images(*self.volume_to_projection(fixed_vol, np.copy(moving_vol), original_=False, transform_fixed=transform_fixed))

        log_ = self.get_transformation_metrics()

        if return_aligned:
            moving_vol_shape = moving_vol.shape
            moving_vol = self.get_ridge_transform(moving_vol)

            if moving_coord is not None:
                moving_coord = self.get_ridge_transform_coord(moving_vol_shape,
                                                              moving_coord,
                                                              *moving_vol.shape[1:])

            return log_, moving_vol, moving_coord
        else:
            return log_


def stitch_tomogram_stack(
    input_dir: str,
    output_dir: str,
    method: str = 'akaze',
    down_scale: int = 10,
):
    """
    Align and stitch a stack of serial-section tomograms from a single folder.

    Workflow:
        1. Discover tomogram images and matching spatial graphs using sort_tomogram_files.
        2. Register each consecutive pair (i → i+1) to obtain a pairwise ridge
           transform (angle, tx, ty, scale).
        3. Accumulate transforms so that every tomogram is expressed in the
           coordinate frame of the first tomogram.
        4. Apply the accumulated transform to each volume and spatial graph.
        5. Stack all transformed volumes and merge all spatial graphs into a
           single output.

    Args:
        input_dir:  Folder containing .am tomogram images and *_spatialGraph.am files.
        output_dir: Folder where the stitched volume and merged spatial graph will
                    be written.
        method:     Registration method forwarded to VolumeRidgeRegistration
                    (default 'akaze').
        down_scale: Downscale factor for the 2-D projection used during
                    registration (default 10).

    Returns:
        A list of per-pair transform dicts [{Angle, Tx, Ty, Scale, Score}, …].
    """
    from os import makedirs
    from os.path import join as pjoin, basename

    logger = logging.getLogger("tardis_em")

    makedirs(output_dir, exist_ok=True)

    # Log file collects all pipeline information
    log_lines = [
        "###############################################################################",
        "# TARDIS - Transformer And Rapid Dimensionless Instance Segmentation (R)      #",
        f"# tardis_em v{version}",
        f"# MIT License * 2021-{datetime.now().year} | Robert Kiewisz & Tristan Bepler",
        "###############################################################################",
        "",
        "--- Settings ---",
        f"Input directory:  {input_dir}",
        f"Output directory: {output_dir}",
        f"Method: {method}",
        f"Down scale: {down_scale}",
        "",
    ]

    # 1. Discover files
    image_paths, coord_paths = sort_tomogram_files(input_dir)
    n = len(image_paths)
    assert n >= 2, f"Need at least 2 tomograms, found {n} in {input_dir}"
    assert len(image_paths) == len(coord_paths)

    logger.info(f"Found {n} tomograms in {input_dir}")

    log_lines.append(f"--- Detected Files ({n} tomograms) ---")
    for i, (ip, cp) in enumerate(zip(image_paths, coord_paths)):
        log_lines.append(f"  [{i}] Image:  {basename(ip)}")
        log_lines.append(f"      Coords: {basename(cp) if cp else 'None'}")
    log_lines.append("")

    # 2. Load all volumes and coords up-front (needed for accumulation)
    volumes = []
    coords_list = []
    pixel_sizes = []

    log_lines.append("--- Loaded Volumes ---")
    for idx, (img_p, coord_p) in enumerate(zip(image_paths, coord_paths)):
        if coord_p is not None and coord_p.endswith(".am"):
            am = ImportDataFromAmira(coord_p, img_p)
            vol, px = am.get_image()
            coord = am.get_segmented_points()
        else:
            vol, px = load_image(img_p)
            coord = None
        volumes.append(vol)
        coords_list.append(coord)
        pixel_sizes.append(px)

        log_lines.append(f"  [{idx}] Shape: {vol.shape}  dtype: {vol.dtype}  "
                         f"px: {px}  coords: {coord.shape if coord is not None else 'None'}")
    log_lines.append("")

    aligner = VolumeRidgeRegistration(method=method, down_scale=down_scale)

    # 3. Compute pairwise transforms and accumulate
    pairwise_transforms = []
    accum_angle = 0.0
    accum_tx = 0.0
    accum_ty = 0.0
    accum_scale = 1.0
    accumulated = [{'Angle': 0.0, 'Tx': 0.0, 'Ty': 0.0, 'Scale': 1.0}]

    log_lines.append("--- Pairwise Registration ---")
    for i in range(n - 1):
        logger.info(f"Registering pair {i} \u2192 {i + 1} ...")

        metric = aligner(volumes[i], volumes[i + 1], return_aligned=False)
        pairwise_transforms.append(metric)

        accum_angle += metric['Angle']
        accum_tx += metric['Tx']
        accum_ty += metric['Ty']
        accum_scale *= metric['Scale']

        accumulated.append({
            'Angle': accum_angle,
            'Tx': accum_tx,
            'Ty': accum_ty,
            'Scale': accum_scale,
        })

        logger.info(f"Pair {i}\u2192{i+1}: Angle={metric['Angle']:.2f}  Tx={metric['Tx']:.1f}  "
                    f"Ty={metric['Ty']:.1f}  Scale={metric['Scale']:.4f}  "
                    f"Score={metric['Score']:.4f}")

        log_lines.append(f"  Pair {i} \u2192 {i+1}:")
        log_lines.append(f"    Fixed:  {basename(image_paths[i])}")
        log_lines.append(f"    Moving: {basename(image_paths[i+1])}")
        log_lines.append(f"    Angle:  {metric['Angle']:.4f}")
        log_lines.append(f"    Tx:     {metric['Tx']:.4f}")
        log_lines.append(f"    Ty:     {metric['Ty']:.4f}")
        log_lines.append(f"    Scale:  {metric['Scale']:.6f}")
        log_lines.append(f"    Score:  {metric['Score']:.6f}")
        log_lines.append("")
    log_lines.append("")

    # 4. Transform each volume and spatial graph with its accumulated transform
    transformed_volumes = []
    transformed_coords = []
    save_am_coord = NumpyToAmira()

    log_lines.append("--- Accumulated Transforms (per tomogram) ---")
    for i in range(n):
        t = accumulated[i]
        logger.info(f"Transforming tomogram {i} (accum: Angle={t['Angle']:.2f}, "
                    f"Tx={t['Tx']:.1f}, Ty={t['Ty']:.1f}, Scale={t['Scale']:.4f})")

        log_lines.append(f"  [{i}] {basename(image_paths[i])}")
        log_lines.append(f"    Angle:  {t['Angle']:.4f}")
        log_lines.append(f"    Tx:     {t['Tx']:.4f}")
        log_lines.append(f"    Ty:     {t['Ty']:.4f}")
        log_lines.append(f"    Scale:  {t['Scale']:.6f}")

        vol = volumes[i]

        if i == 0:
            transformed_volumes.append(vol)
        else:
            aligner.Angle = t['Angle']
            aligner.Tx = t['Tx']
            aligner.Ty = t['Ty']
            aligner.Scale = t['Scale']

            vol_transformed = aligner.get_ridge_transform(vol, reshape=True)
            transformed_volumes.append(vol_transformed)

        log_lines.append(f"    Output shape: {transformed_volumes[i].shape}")
        log_lines.append("")

        # Transform coordinates if available
        coord = coords_list[i]
        if coord is not None and i > 0:
            aligner.Angle = t['Angle']
            aligner.Tx = t['Tx']
            aligner.Ty = t['Ty']
            aligner.Scale = t['Scale']

            vol_shape = volumes[i].shape
            coord_transformed = aligner.get_ridge_transform_coord(
                vol_shape, coord.copy(),
                *transformed_volumes[i].shape[1:]
            )
            transformed_coords.append(coord_transformed)
        else:
            transformed_coords.append(coord)

    # 5. Stack volumes — pad Y/X to the max across all transformed volumes
    max_y = max(v.shape[1] for v in transformed_volumes)
    max_x = max(v.shape[2] for v in transformed_volumes)

    padded_volumes = []
    for v in transformed_volumes:
        dy = max_y - v.shape[1]
        dx = max_x - v.shape[2]
        if dy > 0 or dx > 0:
            v = np.pad(v, ((0, 0), (dy // 2, dy - dy // 2), (dx // 2, dx - dx // 2)),
                       mode='constant', constant_values=0)
        padded_volumes.append(v)

    stitched_vol = np.concatenate(padded_volumes, axis=0)
    logger.info(f"Stitched volume shape: {stitched_vol.shape}")

    # Save stitched volume
    px = pixel_sizes[0] if pixel_sizes[0] is not None else 1.0
    to_am(stitched_vol, px, pjoin(output_dir, 'stitched_volume.am'))
    logger.info(f"Saved stitched volume to {pjoin(output_dir, 'stitched_volume.am')}")

    # 6. Merge spatial graphs — shift Z and re-number segment IDs
    merged_coords = []
    z_offset = 0
    id_offset = 0
    for i, (coord, vol) in enumerate(zip(transformed_coords, transformed_volumes)):
        if coord is not None:
            c = coord.copy()
            c[:, 0] += id_offset
            c[:, 3] += z_offset
            merged_coords.append(c)
            id_offset = int(c[:, 0].max()) + 1
        z_offset += vol.shape[0]

    if merged_coords:
        merged = np.concatenate(merged_coords, axis=0)
        save_am_coord.export_amiraV2(pjoin(output_dir, 'stitched_spatialGraph.am'), merged)
        logger.info(f"Saved merged spatial graph ({merged.shape[0]} points, "
                    f"{int(merged[:, 0].max()) + 1} segments) to "
                    f"{pjoin(output_dir, 'stitched_spatialGraph.am')}")
    else:
        logger.warning("No spatial graphs found to merge.")

    # 7. Write log file
    log_lines.append("--- Stitched Output ---")
    log_lines.append(f"  Volume shape: {stitched_vol.shape}  dtype: {stitched_vol.dtype}")
    log_lines.append(f"  Pixel size:   {px}")
    log_lines.append(f"  Volume file:  {pjoin(output_dir, 'stitched_volume.am')}")
    if merged_coords:
        log_lines.append(f"  SpatialGraph: {pjoin(output_dir, 'stitched_spatialGraph.am')}")
        log_lines.append(f"  Total points:   {merged.shape[0]}")
        log_lines.append(f"  Total segments: {int(merged[:, 0].max()) + 1}")
    else:
        log_lines.append("  SpatialGraph: None (no coordinate files found)")
    log_lines.append("")

    log_path = pjoin(output_dir, "stitch_log.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    logger.info(f"Saved log to {log_path}")

    return pairwise_transforms


def extract_tissue_region(image):
    """Extract tissue region from image as in Kajihara et al."""
    # Convert to grayscale using UV component in YUV
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV) if len(image.shape) == 3 else cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2YUV)
    gray = yuv[:, :, 1]  # U component

    # Binary with Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones_like(binary, dtype=np.uint8) * 255

    # Largest contour
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)

    return mask


def non_rigid_register(source, target, k_max=16):
    """Implement the non-rigid registration from Kajihara et al. 2019"""
    # Convert to uint8 if not
    if source.dtype != np.uint8:
        source = (np.clip(source, 0, 1) * 255).astype(np.uint8)
    if target.dtype != np.uint8:
        target = (np.clip(target, 0, 1) * 255).astype(np.uint8)

    # Convert to grayscale if needed
    if len(source.shape) == 3:
        source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    if len(target.shape) == 3:
        target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    # Extract tissue masks
    mask_source = extract_tissue_region(cv2.cvtColor(source, cv2.COLOR_GRAY2BGR))
    mask_target = extract_tissue_region(cv2.cvtColor(target, cv2.COLOR_GRAY2BGR))

    # AKAZE
    akaze = cv2.AKAZE_create()
    kp_s, des_s = akaze.detectAndCompute(source, mask_source)
    kp_t, des_t = akaze.detectAndCompute(target, mask_target)

    # Match
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(des_s, des_t, k=2)

    # Ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return source  # No registration

    # Points
    pts_s = np.float32([kp_s[m.queryIdx].pt for m in good_matches])
    pts_t = np.float32([kp_t[m.trainIdx].pt for m in good_matches])

    # Determine K
    errors = []
    for k in range(1, k_max + 1):
        if len(pts_s) < k:
            break
        kmeans = KMeans(n_clusters=k, n_init=10)
        labels = kmeans.fit_predict(pts_s)

        transforms = []
        centers = []
        for i in range(k):
            cluster_s = pts_s[labels == i]
            cluster_t = pts_t[labels == i]
            if len(cluster_s) < 4:
                continue
            M, _ = cv2.estimateAffinePartial2D(cluster_s.reshape(-1, 1, 2), cluster_t.reshape(-1, 1, 2), cv2.RANSAC)
            if M is not None:
                transforms.append(M)
                centers.append(np.mean(cluster_s, axis=0))

        if not transforms:
            errors.append(float('inf'))
            continue

        # For simplicity, compute average error (not full DCIB)
        error = 0
        for ps, pt in zip(pts_s, pts_t):
            # Find closest center
            dists = [np.linalg.norm(ps - c) for c in centers]
            idx = np.argmin(dists)
            M = transforms[idx]
            ps_h = np.append(ps, 1)
            pt_pred = M @ ps_h
            error += np.linalg.norm(pt - pt_pred[:2])
        errors.append(error / len(pts_s))

    if not errors:
        return source

    k_opt = np.argmin(errors) + 1

    # Now with k_opt
    kmeans = KMeans(n_clusters=k_opt, n_init=10)
    labels = kmeans.fit_predict(pts_s)

    transforms = []
    centers = []
    for i in range(k_opt):
        cluster_s = pts_s[labels == i]
        cluster_t = pts_t[labels == i]
        if len(cluster_s) >= 4:
            M, _ = cv2.estimateAffinePartial2D(cluster_s.reshape(-1, 1, 2), cluster_t.reshape(-1, 1, 2), cv2.RANSAC)
            if M is not None:
                transforms.append(M)
                centers.append(np.mean(cluster_s, axis=0))

    if not transforms:
        return source

    # For each pixel, find weights and apply average transform (simplified, not DCIB)
    h = int(round(float(source.shape[0])))
    w = int(round(float(source.shape[1])))
    registered = np.zeros_like(source, dtype=np.float32)

    for y in range(h):
        for x in range(w):
            p = np.array([x, y])
            weights = []
            Ms = []
            for c, M in zip(centers, transforms):
                dist = np.linalg.norm(p - c)
                if dist == 0:
                    w = 1
                else:
                    w = 1 / dist**2
                weights.append(w)
                Ms.append(M)
            weights = np.array(weights)
            weights /= weights.sum()

            # Average transform (simplified)
            M_avg = np.zeros((2, 3))
            for w, M in zip(weights, Ms):
                M_avg += w * M

            # Apply
            p_h = np.append(p, 1)
            p_new = M_avg @ p_h
            x_new, y_new = p_new[:2]

            if 0 <= x_new < w and 0 <= y_new < h:
                registered[y, x] = source[int(y_new), int(x_new)]
            else:
                registered[y, x] = 0

    return registered.astype(np.uint8)



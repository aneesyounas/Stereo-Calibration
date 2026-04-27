#!/usr/bin/env python3
import cv2
import glob
import os
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
import time
import json
import cv2.aruco as aruco
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from pathlib import Path
from functools import reduce
from collections import deque
from typing import Optional
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
plt.rcParams.update({'font.size': 16})
import matplotlib.colors as colors
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

per_ccm = True
extrinsic_per_ccm = False

cdict = {
    'red':   ((0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 0.8, 0.8)),
    'green': ((0.0, 0.8, 0.8), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)),
    'blue':  ((0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)),
}
GnRd = colors.LinearSegmentedColormap('GnRd', cdict)


def get_quadrant_coordinates(width, height, nx, ny):
    quadrant_width  = width  // nx
    quadrant_height = height // ny
    quadrant_coords = []
    for i in range(int(nx)):
        for j in range(int(ny)):
            left   = i * quadrant_width
            upper  = j * quadrant_height
            right  = left  + quadrant_width
            bottom = upper + quadrant_height
            quadrant_coords.append((left, upper, right, bottom))
    return quadrant_coords


def sort_points_into_quadrants(points, width, height, error, nx=4, ny=4):
    quadrant_coords = get_quadrant_coordinates(width, height, nx, ny)
    quadrants = {i: [] for i in range(int(nx * ny))}
    for x, y in points:
        for index, (left, upper, right, bottom) in enumerate(quadrant_coords):
            if left <= x < right and upper <= y < bottom:
                quadrants[index].append(error[index])
                break
    return quadrants, quadrant_coords


def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


rectProjectionMode = 0
colors = [(0, 255, 0), (0, 0, 255)]


def setPolygonCoordinates(height, width):
    horizontal_shift = width  // 4
    vertical_shift   = height // 4
    margin = 60
    slope  = 150
    p_coordinates = [
        [[margin, margin], [margin, height-margin],
         [width-margin, height-margin], [width-margin, margin]],
        [[margin, 0], [margin, height],
         [width//2, height-slope], [width//2, slope]],
        [[horizontal_shift, 0], [horizontal_shift, height],
         [width//2+horizontal_shift, height-slope],
         [width//2+horizontal_shift, slope]],
        [[horizontal_shift*2-margin, 0], [horizontal_shift*2-margin, height],
         [width//2+horizontal_shift*2-margin, height-slope],
         [width//2+horizontal_shift*2-margin, slope]],
        [[width-margin, 0], [width-margin, height],
         [width//2, height-slope], [width//2, slope]],
        [[width-horizontal_shift, 0], [width-horizontal_shift, height],
         [width//2-horizontal_shift, height-slope],
         [width//2-horizontal_shift, slope]],
        [[width-horizontal_shift*2+margin, 0], [width-horizontal_shift*2+margin, height],
         [width//2-horizontal_shift*2+margin, height-slope],
         [width//2-horizontal_shift*2+margin, slope]],
        [[0, margin], [width, margin],
         [width-slope, height//2], [slope, height//2]],
        [[0, vertical_shift], [width, vertical_shift],
         [width-slope, height//2+vertical_shift],
         [slope, height//2+vertical_shift]],
        [[0, vertical_shift*2-margin], [width, vertical_shift*2-margin],
         [width-slope, height//2+vertical_shift*2-margin],
         [slope, height//2+vertical_shift*2-margin]],
        [[0, height-margin], [width, height-margin],
         [width-slope, height//2], [slope, height//2]],
        [[0, height-vertical_shift], [width, height-vertical_shift],
         [width-slope, height//2-vertical_shift],
         [slope, height//2-vertical_shift]],
        [[0, height-vertical_shift*2+margin], [width, height-vertical_shift*2+margin],
         [width-slope, height//2-vertical_shift*2+margin],
         [slope, height//2-vertical_shift*2+margin]],
    ]
    return p_coordinates


def getPolygonCoordinates(idx, p_coordinates):
    return p_coordinates[idx]


def getNumOfPolygons(p_coordinates):
    return len(p_coordinates)


def select_polygon_coords(p_coordinates, indexes):
    if indexes is None:
        return p_coordinates
    print("Filtering polygons to those at indexes=", indexes)
    return [p_coordinates[i] for i in indexes]


def image_filename(polygon_index, total_num_of_captured_images):
    return "p{polygon_index}_{total_num_of_captured_images}.png".format(
        polygon_index=polygon_index,
        total_num_of_captured_images=total_num_of_captured_images)


def polygon_from_image_name(image_name):
    import re
    return int(re.findall(r"p(\d+)", image_name)[0])


# ──────────────────────────────────────────────────────────────────────────────
# Shared helper — used in filtering_features AND calibrate_camera_charuco
# ──────────────────────────────────────────────────────────────────────────────
_MIN_CORNERS_FOR_DLT = 6   # cv2 DLT / calibrateCameraCharucoExtended minimum


def _drop_thin_images(corners_list, ids_list, rvecs=None, tvecs=None,
                      label="", min_corners=_MIN_CORNERS_FOR_DLT):
    """
    Remove images that have fewer than min_corners detected corners OR whose
    pose estimation returned None (failed solvePnPRansac).

    Returns (filtered_corners, filtered_ids, n_removed).
    """
    keep = []
    for idx, c in enumerate(corners_list):
        enough_corners = len(c) >= min_corners
        pose_ok = True
        if rvecs is not None and tvecs is not None:
            pose_ok = (rvecs[idx] is not None) and (tvecs[idx] is not None)
        keep.append(enough_corners and pose_ok)

    fc = [c for c, m in zip(corners_list, keep) if m]
    fi = [i for i, m in zip(ids_list,    keep) if m]
    n_removed = sum(1 for m in keep if not m)
    if n_removed:
        print(f"[{label}] Dropped {n_removed} images "
              f"(<{min_corners} corners or failed pose) — "
              f"{len(fc)} images remaining")
    return fc, fi, n_removed


# ──────────────────────────────────────────────────────────────────────────────

class StereoExceptions(Exception):
    def __init__(self, message, stage, path=None, *args, **kwargs):
        self.stage = stage
        self.path  = path
        super().__init__(message, *args, **kwargs)

    @property
    def summary(self):
        return f"'{self.args[0]}' (occured during stage '{self.stage}')"


class StereoCalibration(object):
    """Class to Calculate Calibration and Rectify a Stereo Camera."""

    def __init__(self, traceLevel=1.0, outputScaleFactor=0.5, disableCamera=[],
                 model=None, distortion_model={}, filtering_enable=False,
                 initial_max_threshold=15, initial_min_filtered=0.05,
                 calibration_max_threshold=10, square_size=None):
        self.filtering_enable          = filtering_enable
        self.ccm_model                 = distortion_model
        self.model                     = model
        self.traceLevel                = traceLevel
        self.output_scale_factor       = outputScaleFactor
        self.disableCamera             = disableCamera
        self.errors                    = {}
        self.initial_max_threshold     = initial_max_threshold
        self.initial_min_filtered      = initial_min_filtered
        self.calibration_max_threshold = calibration_max_threshold
        self.calibration_min_filtered  = initial_min_filtered
        self.square_size_cm            = square_size

    # ──────────────────────────────────────────────────────────────────────────

    def calibrate(self, board_config, filepath, square_size, mrk_size,
                  squaresX, squaresY, camera_model, enable_disp_rectify,
                  charucos={}, intrinsic_img={}, extrinsic_img=[]):

        start_time = time.time()
        if self.traceLevel in (2, 10):
            print(f'squareX is {squaresX}')

        self.enable_rectification_disp = True
        if intrinsic_img != {}:
            for cam in intrinsic_img: intrinsic_img[cam].sort(reverse=True)
        if extrinsic_img != {}:
            for cam in extrinsic_img: extrinsic_img[cam].sort(reverse=True)

        self.intrinsic_img   = intrinsic_img
        self.extrinsic_img   = extrinsic_img
        self.cameraModel     = camera_model
        self.cameraIntrinsics   = {}
        self.cameraDistortion   = {}
        self.distortion_model   = {}
        self.calib_model        = {}
        self.collected_features = {}
        self.collected_ids      = {}
        self.all_features       = {}
        self.all_errors         = {}
        self.errors             = {}
        self.data_path          = filepath
        self.charucos           = charucos
        self.aruco_dictionary   = aruco.Dictionary_get(aruco.DICT_4X4_1000)
        self.squaresX           = squaresX
        self.squaresY           = squaresY

        if mrk_size is None or mrk_size <= 0:
            print("Plain chessboard mode — no Charuco board created")
            self.board = None
            self.is_chessboard_mode = True
        else:
            print("Charuco mode detected")
            self.board = aruco.CharucoBoard_create(
                squaresX, squaresY, square_size, mrk_size, self.aruco_dictionary)
            self.is_chessboard_mode = False

        self.cams            = []
        combinedCoverageImage = None
        resizeWidth, resizeHeight = 1280, 800
        self.height = {}
        self.width  = {}

        if mrk_size is not None and mrk_size > 0:
            assert mrk_size > 0, "Marker size must be positive for Charuco mode"

        # ── measure image sizes ───────────────────────────────────────────────
        for camera in board_config['cameras'].keys():
            cam_info = board_config['cameras'][camera]
            if cam_info["name"] not in self.disableCamera:
                images_path = filepath + '/' + cam_info['name']
                image_files = glob.glob(images_path + "/*")
                image_files.sort()
                for im in image_files:
                    frame = cv2.imread(im)
                    self.height[cam_info["name"]], self.width[cam_info["name"]], _ = frame.shape
                    wR = resizeWidth  / self.width[cam_info["name"]]
                    hR = resizeHeight / self.height[cam_info["name"]]
                    if ((wR > 0.8 and hR > 0.8 and wR <= 1.0 and hR <= 1.0)
                            or (wR > 1.2 and hR > 1.2) or resizeHeight == 0):
                        resizeWidth  = self.width[cam_info["name"]]
                        resizeHeight = self.height[cam_info["name"]]
                    break

        # ── intrinsic calibration per camera ──────────────────────────────────
        for camera in board_config['cameras'].keys():
            cam_info = board_config['cameras'][camera]
            self.id  = cam_info["name"]
            if cam_info["name"] not in self.disableCamera:
                print('<------------Calibrating {} ------------>'.format(cam_info['name']))
                images_path    = filepath + '/' + cam_info['name']
                distCoeffsInit = np.zeros((12, 1))

                if "calib_model" in cam_info.keys():
                    self.cameraModel_ccm, self.model_ccm = cam_info["calib_model"].split("_")
                    if self.cameraModel_ccm == "fisheye":
                        distCoeffsInit = np.zeros((4, 1))
                    self.calib_model[cam_info["name"]]      = self.cameraModel_ccm
                    self.distortion_model[cam_info["name"]] = self.model_ccm
                else:
                    self.calib_model[cam_info["name"]] = self.cameraModel
                    if cam_info["name"] in self.ccm_model:
                        self.distortion_model[cam_info["name"]] = self.ccm_model[cam_info["name"]]
                    else:
                        self.distortion_model[cam_info["name"]] = self.model

                features       = None
                self.img_path  = glob.glob(images_path + "/*")
                if charucos == {}:
                    try:
                        self.img_path = sorted(self.img_path, key=lambda x: int(x.split('_')[1]))
                    except:
                        self.img_path.sort()
                else:
                    self.img_path.sort()

                cam_info["img_path"] = self.img_path
                self.name = cam_info["name"]

                if per_ccm:
                    all_features, all_ids, imsize = self.getting_features(
                        images_path, cam_info["name"], features=features)
                    if isinstance(all_features, str) and all_ids is None:
                        self.errors.setdefault(cam_info["name"], []).append(all_features)
                        continue

                    cam_info["imsize"] = imsize
                    f = imsize[0] / (2 * np.tan(np.deg2rad(cam_info["hfov"] / 2)))
                    print("INTRINSIC CALIBRATION")
                    cameraMatrixInit = np.array([[f, 0., imsize[0]/2],
                                                 [0., f, imsize[1]/2],
                                                 [0., 0., 1.]])
                    self.cameraIntrinsics.setdefault(cam_info["name"], cameraMatrixInit)
                    if self.traceLevel in (3, 10):
                        print(f'Camera Matrix init for {cam_info["name"]}:')
                        print(cameraMatrixInit)
                    self.cameraDistortion.setdefault(cam_info["name"], distCoeffsInit)

                    filtered_images = images_path
                    current_time    = time.time()

                    if self.cameraModel != "fisheye":
                        print("Filtering corners")
                        if self.is_chessboard_mode:
                            print("[Chessboard] Using original corners/ids (no advanced filtering)")
                            removed_features  = []
                            filtered_features = all_features
                            filtered_ids      = all_ids
                        else:
                            removed_features, filtered_features, filtered_ids = \
                                self.filtering_features(
                                    all_features, all_ids, cam_info["name"], imsize,
                                    cam_info["hfov"], cameraMatrixInit, distCoeffsInit)

                        if filtered_features is None:
                            self.errors.setdefault(cam_info["name"], []).append(removed_features)
                            continue
                        print(f"Filtering takes: {time.time()-current_time}")
                        self.collected_features.setdefault(cam_info["name"], filtered_features)
                        self.collected_ids.setdefault(cam_info["name"], filtered_ids)
                    else:
                        filtered_features = all_features
                        filtered_ids      = all_ids

                    cam_info['filtered_ids']     = filtered_ids
                    cam_info['filtered_corners'] = filtered_features

                    ret, intrinsics, dist_coeff, _, _, filtered_ids, filtered_corners, \
                        size, coverageImage, all_corners, all_ids = \
                        self.calibrate_wf_intrinsics(
                            cam_info["name"], all_features, all_ids,
                            filtered_features, filtered_ids,
                            cam_info["imsize"], cam_info["hfov"], features, filtered_images)

                    if isinstance(ret, str) and all_ids is None:
                        self.errors.setdefault(cam_info["name"], []).append(ret)
                        continue
                else:
                    ret, intrinsics, dist_coeff, _, _, filtered_ids, filtered_corners, \
                        size, coverageImage, all_corners, all_ids = \
                        self.calibrate_intrinsics(images_path, cam_info['hfov'], cam_info["name"])
                    cam_info['filtered_ids']     = filtered_ids
                    cam_info['filtered_corners'] = filtered_corners

                self.cameraIntrinsics[cam_info["name"]] = intrinsics
                self.cameraDistortion[cam_info["name"]] = dist_coeff
                cam_info['intrinsics']        = intrinsics
                cam_info['dist_coeff']        = dist_coeff
                cam_info['size']              = size
                cam_info['reprojection_error']= ret
                print("Reprojection error of {0}: {1}".format(cam_info['name'], ret))
                if self.traceLevel in (3, 10):
                    print("Estimated intrinsics of {0}: \n {1}".format(cam_info['name'], intrinsics))

                # coverage image
                coverage_name = cam_info['name']
                print_text = (f'Coverage Image of {coverage_name} with reprojection error '
                              f'of {round(ret, 5)}')
                h, w, _ = coverageImage.shape
                if w > resizeWidth and h > resizeHeight:
                    coverageImage = cv2.resize(coverageImage, (0,0),
                                               fx=resizeWidth/w, fy=resizeWidth/w)
                h, w, _ = coverageImage.shape
                if h > resizeHeight:
                    ho = (h - resizeHeight) // 2
                    coverageImage = coverageImage[ho:ho+resizeHeight, :]
                h, w, _ = coverageImage.shape
                ho = (resizeHeight - h) // 2
                wo = (resizeWidth  - w) // 2
                subImage = np.pad(coverageImage,
                                  ((ho, ho), (wo, wo), (0, 0)),
                                  'constant', constant_values=0)
                cv2.putText(subImage, print_text, (50, 50+ho),
                            cv2.FONT_HERSHEY_SIMPLEX, 2*coverageImage.shape[0]/1750, (0,0,0), 2)
                combinedCoverageImage = (subImage if combinedCoverageImage is None
                                         else np.hstack((combinedCoverageImage, subImage)))
                cv2.imwrite(filepath + '/' + coverage_name + '_coverage.png', subImage)

        if self.errors:
            raise StereoExceptions(
                message="".join(v[0]+"\n" for v in self.errors.values()),
                stage="intrinsic")

        combinedCoverageImage = cv2.resize(
            combinedCoverageImage, (0,0),
            fx=self.output_scale_factor, fy=self.output_scale_factor)
        if enable_disp_rectify:
            cv2.waitKey(1)
            cv2.destroyAllWindows()

        # ── extrinsic / stereo calibration ────────────────────────────────────
        for camera in board_config['cameras'].keys():
            left_cam_info = board_config['cameras'][camera]
            if str(left_cam_info["name"]) not in self.disableCamera:
                if 'extrinsics' in left_cam_info and 'to_cam' in left_cam_info['extrinsics']:
                    left_cam  = camera
                    right_cam = left_cam_info['extrinsics']['to_cam']
                    right_cam_info = board_config['cameras'][right_cam]

                    if str(right_cam_info["name"]) not in self.disableCamera:
                        print('<-------------Extrinsics calibration of {} and {} ------------>'.format(
                            left_cam_info['name'], right_cam_info['name']))

                        specT = left_cam_info['extrinsics']['specTranslation']
                        rot   = left_cam_info['extrinsics']['rotation']
                        translation = np.array(
                            [specT['x'], specT['y'], specT['z']], dtype=np.float32)
                        rotation = Rotation.from_euler(
                            'xyz', [rot['r'], rot['p'], rot['y']], degrees=True
                        ).as_matrix().astype(np.float32)

                        extrinsics = self.calibrate_stereo(
                            left_cam_info['name'],  right_cam_info['name'],
                            left_cam_info['filtered_ids'],     left_cam_info['filtered_corners'],
                            right_cam_info['filtered_ids'],    right_cam_info['filtered_corners'],
                            left_cam_info['intrinsics'],       left_cam_info['dist_coeff'],
                            right_cam_info['intrinsics'],      right_cam_info['dist_coeff'],
                            translation, rotation, features)

                        if extrinsics[0] == -1:
                            return -1, extrinsics[1]

                        sc = board_config['stereo_config']
                        if sc['left_cam'] == left_cam and sc['right_cam'] == right_cam:
                            sc['rectification_left']  = extrinsics[3]
                            sc['rectification_right'] = extrinsics[4]
                        elif sc['left_cam'] == right_cam and sc['right_cam'] == left_cam:
                            sc['rectification_left']  = extrinsics[4]
                            sc['rectification_right'] = extrinsics[3]

                        print('<-------------Epipolar error of {} and {} ------------>'.format(
                            left_cam_info['name'], right_cam_info['name']))

                        if per_ccm and extrinsic_per_ccm:
                            scale = ((left_cam_info['intrinsics'][0][0]
                                      * right_cam_info['intrinsics'][0][0]
                                      + left_cam_info['intrinsics'][1][1]
                                      * right_cam_info['intrinsics'][1][1]) / 2)
                            epi = extrinsics[0] * np.sqrt(scale)
                        else:
                            epi = extrinsics[0]
                        print(f"Epipolar error {epi}")
                        left_cam_info['extrinsics']['epipolar_error'] = epi
                        left_cam_info['extrinsics']['stereo_error']   = epi
                        left_cam_info['extrinsics']['rotation_matrix']= extrinsics[1]
                        left_cam_info['extrinsics']['translation']    = extrinsics[2]

        return 1, board_config

    # ──────────────────────────────────────────────────────────────────────────
    def getting_features(self, img_path, name, features=None):
        if self.charucos != {}:
            allCorners, allIds = [], []
            for ids, charucos in self.charucos[name]:
                allCorners.append(charucos); allIds.append(ids)
            return allCorners, allIds, (self.width[name], self.height[name])

        img_files = glob.glob(img_path + "/*")
        if not img_files:
            return f"No images found in {img_path}", None, None
        img_files.sort()
        allCorners, allIds = [], []
        imsize = None

        if self.is_chessboard_mode:
            print(f"Processing plain chessboard images for camera: {name}")
            board_size = (self.squaresX, self.squaresY)
            for im in img_files:
                img = cv2.imread(im)
                if img is None: continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if imsize is None: imsize = gray.shape[::-1]
                ret, corners = cv2.findChessboardCorners(
                    gray, board_size,
                    flags=(cv2.CALIB_CB_ADAPTIVE_THRESH
                           + cv2.CALIB_CB_FAST_CHECK
                           + cv2.CALIB_CB_NORMALIZE_IMAGE))
                if ret:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                    corners = corners.reshape(-1, 1, 2)
                    allCorners.append(corners)
                    allIds.append(np.arange(len(corners)).reshape(-1,1).astype(np.int32))
                    print(f"Chessboard found in {os.path.basename(im)} - {len(corners)} corners")
                else:
                    print(f"Chessboard NOT found in {os.path.basename(im)}")
            if not allCorners:
                return f"No valid chessboard detections for {name}", None, None
            print(f"Total valid images: {len(allCorners)} / {len(img_files)}")
            return allCorners, allIds, imsize
        else:
            print(f"Processing Charuco board images for camera: {name}")
            allCorners, allIds, _, _, imsize, _ = self.analyze_charuco(img_files)
            if isinstance(allCorners, str):
                return allCorners, None, None
            return allCorners, allIds, imsize

    # ──────────────────────────────────────────────────────────────────────────
    def filtering_features(self, allCorners, allIds, name, imsize, hfov,
                           cameraMatrixInit, distCoeffsInit):
        if self.board is None:
            print(f"Skipping advanced filtering in chessboard mode for {name}")
            return allCorners, allIds, []

        max_threshold    = 75 + self.initial_max_threshold * (hfov/30 + imsize[1]/800*0.2)
        threshold_stepper= int(1.5 * (hfov/30 + imsize[1]/800))
        if threshold_stepper < 1: threshold_stepper = 1
        print(threshold_stepper)
        min_inlier = 1 - self.initial_min_filtered * (hfov/60 + imsize[1]/800*0.2)

        # ── pose estimation ───────────────────────────────────────────────────
        for index, corners in enumerate(allCorners):
            if len(corners) < 4:
                return f"Less than 4 corners detected on image {index}.", None, None

        rvecs, tvecs = [], []
        overall_pose = time.time()
        self.index   = 0
        for index, (corners, ids) in enumerate(zip(allCorners, allIds)):
            self.index = index
            t0 = time.time()
            objpts = self.charuco_ids_to_objpoints(ids)
            rvec, tvec, _ = self.camera_pose_charuco(
                objpts, corners, ids, cameraMatrixInit, distCoeffsInit,
                max_threshold=max_threshold, min_inliers=min_inlier,
                ini_threshold=5, threshold_stepper=threshold_stepper)
            tvecs.append(tvec)
            rvecs.append(rvec)
            print(f"Pose estimation {index}, {time.time()-t0}s")
        print(f"Overall pose estimation {time.time()-overall_pose}s")

        # feature filtering (bypassed — returns original corners)
        ret = 0.0
        distortion_flags = self.get_distortion_flags(name)
        flags = cv2.CALIB_USE_INTRINSIC_GUESS + distortion_flags
        t0    = time.time()
        filtered_corners, filtered_ids, all_error, _, _, _ = \
            self.features_filtering_function(
                rvecs, tvecs, cameraMatrixInit, distCoeffsInit,
                ret, allCorners, allIds, camera=name)

        good_count    = len([c for c in filtered_corners if len(c) >= 4])
        total_images  = len(self.img_path)
        percent_good  = good_count / total_images if total_images > 0 else 0
        if percent_good < 0.005:
            print(f"WARNING: Only {good_count}/{total_images} ({percent_good:.1%}) "
                  f"images ≥4 corners for {name} — proceeding anyway")
        else:
            print(f"Keeping {good_count}/{total_images} good images ({percent_good:.1%}) "
                  f"for {name}")
        print(f"Filtering {time.time()-t0}s")

        # ══════════════════════════════════════════════════════════════════════
        # FIX — Apply ≥6 corner guard AND pose-None guard HERE, right before
        # the first calibrateCameraCharucoExtended call.
        # This is the actual crash site: "First intrinsic calibration failed"
        # ══════════════════════════════════════════════════════════════════════
        filtered_corners, filtered_ids, n_dropped = _drop_thin_images(
            filtered_corners, filtered_ids,
            rvecs=rvecs, tvecs=tvecs,
            label=name)
        if not filtered_corners:
            return (f"No images with ≥{_MIN_CORNERS_FOR_DLT} corners / valid pose "
                    f"for {name} after filtering"), None, None
        # ══════════════════════════════════════════════════════════════════════

        if self.board is not None:
            try:
                (ret, camera_matrix, distortion_coefficients,
                 rotation_vectors, translation_vectors,
                 stdDeviationsIntrinsics, stdDeviationsExtrinsics,
                 perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                    charucoCorners=filtered_corners,
                    charucoIds=filtered_ids,
                    board=self.board,
                    imageSize=imsize,
                    cameraMatrix=cameraMatrixInit,
                    distCoeffs=distCoeffsInit,
                    flags=flags,
                    criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 1000, 1e-6))
            except Exception as e:
                print(f"[{name}] calibrateCameraCharucoExtended in filtering_features "
                      f"failed: {e}")
                return (f"First intrinsic calibration failed for {name}", None, None)
            self.cameraIntrinsics[name] = camera_matrix
            self.cameraDistortion[name] = distortion_coefficients

        return [], filtered_corners, filtered_ids

    # ──────────────────────────────────────────────────────────────────────────
    def remove_features(self, allCorners, allIds, array, img_files=None):
        fc = allCorners.copy()
        fi = allIds.copy()
        fp = img_files.copy() if img_files is not None else None
        for index in sorted(array, reverse=True):
            fc.pop(index); fi.pop(index)
            if fp is not None: fp.pop(index)
        return fc, fi, fp

    # ──────────────────────────────────────────────────────────────────────────
    def get_distortion_flags(self, name):
        def is_bin(s): return all(ch in '01' for ch in s)
        dm = self.distortion_model[name]
        if dm is None:
            print("Use DEFAULT model")
            return cv2.CALIB_RATIONAL_MODEL
        if isinstance(dm, str) and is_bin(dm):
            flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_TILTED_MODEL + cv2.CALIB_THIN_PRISM_MODEL
            bn    = int(dm, 2)
            cs    = [True]*9 if bn == 0 else [(bn & (1<<i))!=0 for i in range(len(dm))][::-1]
            lut   = [cv2.CALIB_FIX_K1, cv2.CALIB_FIX_K2, cv2.CALIB_FIX_K3,
                     cv2.CALIB_FIX_K4, cv2.CALIB_FIX_K5, cv2.CALIB_FIX_K6,
                     cv2.CALIB_ZERO_TANGENT_DIST, cv2.CALIB_FIX_TAUX_TAUY,
                     cv2.CALIB_FIX_S1_S2_S3_S4]
            names = ["FIX_K1","FIX_K2","FIX_K3","FIX_K4","FIX_K5","FIX_K6",
                     "FIX_TANGENT","FIX_TILTED","FIX_PRISM"]
            for i, (c, lf, ln) in enumerate(zip(cs, lut, names)):
                if c: print(ln); flags += lf
            return flags
        if isinstance(dm, str):
            base = cv2.CALIB_RATIONAL_MODEL
            if dm in ("NORMAL", "TILTED"): return base + cv2.CALIB_TILTED_MODEL
            if dm == "PRISM":   return base + cv2.CALIB_TILTED_MODEL + cv2.CALIB_THIN_PRISM_MODEL
            if dm == "THERMAL": return base + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6
        if isinstance(dm, int):
            print("Using CUSTOM flags"); return dm
        print("Use DEFAULT model"); return cv2.CALIB_RATIONAL_MODEL

    def get_fisheye_distortion_flags(self, name):
        def is_bin(s): return all(ch in '01' for ch in s)
        dm = self.distortion_model[name]
        if dm is None: return cv2.CALIB_RATIONAL_MODEL
        if isinstance(dm, str) and is_bin(dm):
            flags = cv2.CALIB_RATIONAL_MODEL; bn = int(dm, 2)
            cs    = [True]*4 if bn==0 else [(bn&(1<<i))!=0 for i in range(len(dm))][::-1]
            lut   = [cv2.fisheye.CALIB_FIX_K1, cv2.fisheye.CALIB_FIX_K2,
                     cv2.fisheye.CALIB_FIX_K3, cv2.fisheye.CALIB_FIX_K4]
            for c, lf in zip(cs, lut):
                if c: flags += lf
            return flags
        if isinstance(dm, int): return dm
        return cv2.CALIB_RATIONAL_MODEL

    # ──────────────────────────────────────────────────────────────────────────
    def calibrate_wf_intrinsics(self, name, all_Features, all_features_Ids,
                                allCorners, allIds, imsize, hfov, features, image_files):
        image_files = glob.glob(image_files + "/*"); image_files.sort()
        coverageImage = np.ones(imsize[::-1], np.uint8) * 255
        coverageImage = cv2.cvtColor(coverageImage, cv2.COLOR_GRAY2BGR)
        coverageImage = self.draw_corners(allCorners, coverageImage)

        if self.calib_model[name] == 'perspective':
            if features is None or features == "charucos":
                distortion_flags = self.get_distortion_flags(name)
                ret, camera_matrix, dist_coeff, rv, tv, fid, fc, ac, ai = \
                    self.calibrate_camera_charuco(
                        all_Features, all_features_Ids,
                        allCorners, allIds, imsize, hfov, name, distortion_flags)
            if self.charucos == {}:
                self.undistort_visualization(image_files, camera_matrix, dist_coeff, imsize, name)
            return ret, camera_matrix, dist_coeff, rv, tv, fid, fc, imsize, coverageImage, ac, ai
        else:
            print('Fisheye--------------------------------------------------')
            ret, camera_matrix, dist_coeff, rv, tv, fid, fc = \
                self.calibrate_fisheye(allCorners, allIds, imsize, hfov, name)
            self.undistort_visualization(image_files, camera_matrix, dist_coeff, imsize, name)
            return ret, camera_matrix, dist_coeff, rv, tv, fid, fc, imsize, coverageImage, allCorners, allIds

    # ──────────────────────────────────────────────────────────────────────────
    def draw_corners(self, charuco_corners, displayframe):
        for corners in charuco_corners:
            color = tuple(int(np.random.randint(0,255)) for _ in range(3))
            for corner in corners:
                cv2.circle(displayframe, (int(corner[0][0]), int(corner[0][1])), 4, color, -1)
        h, w = displayframe.shape[:2]
        cv2.line(displayframe, (0,0), (0,h), (0,0,0), 4)
        return displayframe

    # ──────────────────────────────────────────────────────────────────────────
    def features_filtering_function(self, rvecs, tvecs, cameraMatrix, distCoeffs,
                                    reprojection, filtered_corners, filtered_id,
                                    camera, display=True, threshold=None,
                                    draw_quadrants=False, nx=4, ny=4):
        # Bypassed — returns all corners unchanged so the main calibration loop
        # gets to run its own iterative solver.
        print("*** BYPASSING FILTERING — returning original corners ***")
        dummy = [0.0] * sum(len(c) for c in filtered_corners)
        return filtered_corners, filtered_id, dummy, [], [], []

    # ──────────────────────────────────────────────────────────────────────────
    def detect_charuco_board(self, image):
        arucoParams = cv2.aruco.DetectorParameters_create()
        arucoParams.minMarkerDistanceRate = 0.01
        corners, ids, rejected = cv2.aruco.detectMarkers(
            image, self.aruco_dictionary, parameters=arucoParams)
        mc, mi, _, _ = cv2.aruco.refineDetectedMarkers(
            image, self.board, corners, ids, rejectedCorners=rejected)
        if len(mc) > 0:
            ret, cc, ci = cv2.aruco.interpolateCornersCharuco(mc, mi, image, self.board, minMarkers=1)
            return ret, cc, ci, mc, mi
        return None, None, None, None, None

    # ──────────────────────────────────────────────────────────────────────────
    def camera_pose_charuco(self, objpoints, corners, ids, K, d,
                            ini_threshold=2, min_inliers=0.95,
                            threshold_stepper=1, max_threshold=50):
        objects = []
        index   = 0
        while len(objects) < len(objpoints) * min_inliers:
            if ini_threshold > max_threshold:
                break
            ret, rvec, tvec, objects_raw = cv2.solvePnPRansac(
                objpoints, corners, K, d,
                flags=cv2.SOLVEPNP_P3P,
                reprojectionError=ini_threshold,
                iterationsCount=10000, confidence=0.9)
            objects = [] if objects_raw is None else objects_raw.ravel().tolist()
            if len(objects) == 0:
                ini_threshold += threshold_stepper; index += 1; continue
            inlier_mask = np.zeros(len(corners), dtype=bool)
            inlier_mask[np.array(objects).astype(int)] = True
            ret, rvec, tvec = cv2.solvePnP(
                objpoints[inlier_mask], corners[inlier_mask], K, d)
            ini_threshold += threshold_stepper; index += 1
        if ret and len(objects) > 0:
            return rvec, tvec, np.array(objects).reshape(-1,1)
        return None, None, None

    # ──────────────────────────────────────────────────────────────────────────
    def compute_reprojection_errors(self, obj_pts, img_pts, K, dist, rvec, tvec, fisheye=False):
        fn = cv2.fisheye.projectPoints if fisheye else cv2.projectPoints
        proj_pts, _ = fn(obj_pts, rvec, tvec, K, dist)
        return np.linalg.norm(np.squeeze(proj_pts) - np.squeeze(img_pts), axis=1)

    # ──────────────────────────────────────────────────────────────────────────
    def charuco_ids_to_objpoints(self, ids):
        if self.board is None:
            sq = (self.square_size_cm / 100.0) if self.square_size_cm is not None else 0.03
            nx, ny = self.squaresX, self.squaresY
            pts = []
            for i in range(len(ids)):
                idx = int(ids[i][0])
                pts.append([idx % nx * sq, idx // nx * sq, 0.0])
            return np.array(pts, dtype=np.float32)
        one_pts = self.board.chessboardCorners
        return np.array([one_pts[ids[j]] for j in range(len(ids))])

    # ──────────────────────────────────────────────────────────────────────────
    def analyze_charuco(self, images, scale_req=False, req_resolution=(800, 1280)):
        allCorners, allIds = [], []
        all_mc, all_mi, all_rec = [], [], []
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.00001)
        skip_vis = False
        gray = None

        for im in images:
            if self.traceLevel in (3, 10): print(f"=> Processing {im}")
            frame = cv2.imread(im)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if scale_req:
                exp_h = gray.shape[0] * (req_resolution[1] / gray.shape[1])
                if not (gray.shape[0] == req_resolution[0]
                        and gray.shape[1] == req_resolution[1]):
                    if int(exp_h) == req_resolution[0]:
                        gray = cv2.resize(gray, req_resolution[::-1], interpolation=cv2.INTER_CUBIC)
                    else:
                        sw = req_resolution[1] / gray.shape[1]
                        gray = cv2.resize(gray, (int(gray.shape[1]*sw), int(gray.shape[0]*sw)),
                                          interpolation=cv2.INTER_CUBIC)
                        dh = (gray.shape[0] - req_resolution[0]) // 2
                        gray = gray[dh:dh+req_resolution[0], :]

            ret, cc, ci, mc, mi = self.detect_charuco_board(gray)
            if cc is not None and ci is not None and len(cc) > 3:
                cc = cv2.cornerSubPix(gray, cc, (5,5), (-1,-1), criteria)
                allCorners.append(cc); allIds.append(ci)
                all_mc.append(mc);    all_mi.append(mi)
            else:
                n = len(cc) if cc is not None else 0
                print(f"SKIP: only {n} corners detected — ignoring {im}")

        if gray is None:
            return "No images processed", None, None, None, None, None
        return allCorners, allIds, all_mc, all_mi, gray.shape[::-1], all_rec

    # ──────────────────────────────────────────────────────────────────────────
    def calibrate_intrinsics(self, image_files, hfov, name):
        image_files = glob.glob(image_files + "/*"); image_files.sort()
        assert image_files, "ERROR: Images not found"
        allCorners, allIds, _, _, imsize, _ = (
            self.analyze_charuco(image_files) if self.charucos == {}
            else ([], [], None, None, (self.height[name], self.width[name]), None))
        if self.charucos != {}:
            allCorners, allIds = [], []
            for ids, charucos in self.charucos[name]:
                allCorners.append(charucos); allIds.append(ids)
        coverageImage = np.ones(imsize[::-1], np.uint8) * 255
        coverageImage = cv2.cvtColor(coverageImage, cv2.COLOR_GRAY2BGR)
        coverageImage = self.draw_corners(allCorners, coverageImage)
        if self.calib_model[name] == 'perspective':
            df  = self.get_distortion_flags(name)
            ret, K, D, rv, tv, fi, fc, ac, ai = self.calibrate_camera_charuco(
                allCorners, allIds, imsize, hfov, name, df)
            self.undistort_visualization(image_files, K, D, imsize, name)
            return ret, K, D, rv, tv, fi, fc, imsize, coverageImage, ac, ai
        else:
            ret, K, D, rv, tv, fi, fc = self.calibrate_fisheye(allCorners, allIds, imsize, hfov, name)
            self.undistort_visualization(image_files, K, D, imsize, name)
            return ret, K, D, rv, tv, fi, fc, imsize, coverageImage, allCorners, allIds

    # ──────────────────────────────────────────────────────────────────────────
    def scale_intrinsics(self, intrinsics, originalShape, destShape):
        scale = destShape[1] / originalShape[1]
        sm    = np.array([[scale,0,0],[0,scale,0],[0,0,1]])
        si    = np.matmul(sm, intrinsics)
        si[1][2] -= (originalShape[0]*scale - destShape[0]) / 2
        si[0][2] -= (originalShape[1]*scale - destShape[1]) / 2
        return si

    # ──────────────────────────────────────────────────────────────────────────
    def undistort_visualization(self, img_list, K, D, img_size, name):
        for index, im in enumerate(img_list):
            img = cv2.imread(im)
            if self.cameraModel == 'perspective':
                kS, _ = cv2.getOptimalNewCameraMatrix(K, D, img_size, 0)
                m1, m2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), kS, img_size, cv2.CV_32FC1)
            else:
                m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, img_size, cv2.CV_32FC1)
            und = cv2.remap(img, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            if index == 0:
                cv2.imwrite(self.data_path + '/' + name + '_undistorted.png', und)
            if self.traceLevel in (4, 5, 10):
                cv2.putText(und, "Press S to save", (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                            2*und.shape[0]/1750, (0,0,255), 2)
                cv2.imshow("undistorted", und)
                k = cv2.waitKey(1)
                if k in (ord('s'), ord('S')):
                    cv2.imwrite(self.data_path+'/'+name+f'_{index}_undistorted.png', und)
                if k == 27: break
                cv2.destroyWindow("undistorted")

    # ──────────────────────────────────────────────────────────────────────────
    def filter_corner_outliers(self, allIds, allCorners, K, D, rvecs, tvecs):
        corners_removed = False
        for i in range(len(allIds)):
            objpts = self.charuco_ids_to_objpoints(allIds[i])
            errs   = self.compute_reprojection_errors(
                objpts, allCorners[i], K, D, rvecs[i], tvecs[i],
                fisheye=(self.cameraModel=="fisheye"))
            thr = max(2*np.median(errs), 100)
            bad = np.where(errs > thr)[0]
            if 0 < len(bad) < len(allCorners[i])/5:
                corners_removed = True
                allCorners[i] = np.delete(allCorners[i], bad, axis=0)
                allIds[i]     = np.delete(allIds[i], bad, axis=0)
        return corners_removed, allIds, allCorners

    # ──────────────────────────────────────────────────────────────────────────
    def calibrate_camera_charuco(self, all_Features, all_features_Ids,
                                 allCorners, allIds, imsize, hfov, name, distortion_flags):
        # ── plain chessboard branch ───────────────────────────────────────────
        if self.board is None:
            sq = (self.square_size_cm/100.) if self.square_size_cm else 0.03
            f  = imsize[0] / (2*np.tan(np.deg2rad(hfov/2)))
            Ki = np.array([[f,0.,imsize[0]/2],[0.,f,imsize[1]/2],[0.,0.,1.]], dtype=np.float32)
            Di = np.zeros((5,1), dtype=np.float32)
            nx, ny = self.squaresX, self.squaresY
            objp = np.zeros((nx*ny,3), np.float32)
            objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) * sq
            op, ip = [], []
            for corners in allCorners: op.append(objp); ip.append(corners.reshape(-1,2))
            flags = cv2.CALIB_USE_INTRINSIC_GUESS + distortion_flags
            ret, K, D, rv, tv = cv2.calibrateCamera(op, ip, imsize, Ki, Di, flags=flags)
            return ret, K, D, rv, tv, allIds, allCorners, allCorners, allIds

        # ── Charuco branch ────────────────────────────────────────────────────
        f = imsize[0] / (2*np.tan(np.deg2rad(hfov/2)))
        if name not in self.cameraIntrinsics:
            cameraMatrixInit = np.array([[f,0.,imsize[0]/2],[0.,f,imsize[1]/2],[0.,0.,1.]])
            threshold        = 20 * imsize[1]/800.
        else:
            cameraMatrixInit = self.cameraIntrinsics[name]
            threshold        = 2  * imsize[1]/800.

        distCoeffsInit = (np.zeros((5,1)) if name not in self.cameraDistortion
                          else self.cameraDistortion[name])

        # ══════════════════════════════════════════════════════════════════════
        # FIX — ≥6 corner guard (defence-in-depth for the main iterative loop)
        # Primary guard is in filtering_features; this catches any stragglers.
        # ══════════════════════════════════════════════════════════════════════
        allCorners, allIds, _ = _drop_thin_images(
            allCorners, allIds, label=f"{name}[calibrate_camera_charuco]")
        all_Features, all_features_Ids, _ = _drop_thin_images(
            all_Features, all_features_Ids, label=f"{name}[calibrate_camera_charuco/all]")
        if not allCorners:
            return (f"No images with ≥{_MIN_CORNERS_FOR_DLT} corners in "
                    f"calibrate_camera_charuco for {name}",
                    None, None, None, None, None, None, None, None)
        # ══════════════════════════════════════════════════════════════════════

        rvecs, tvecs = [], []
        self.index   = 0
        for index, (corners, ids) in enumerate(zip(allCorners, allIds)):
            self.index = index
            objpts = self.charuco_ids_to_objpoints(ids)
            rv, tv, _ = self.camera_pose_charuco(objpts, corners, ids,
                                                 cameraMatrixInit, distCoeffsInit)
            tvecs.append(tv); rvecs.append(rv)

        ret = 0.0
        flags = cv2.CALIB_USE_INTRINSIC_GUESS + distortion_flags
        camera_matrix           = cameraMatrixInit
        distortion_coefficients = distCoeffsInit
        rotation_vectors        = rvecs
        translation_vectors     = tvecs
        previous_ids            = []
        index                   = 0

        try:
            whole = time.time()
            while True:
                t0 = time.time()
                filtered_corners, filtered_ids, _, _, _, _ = \
                    self.features_filtering_function(
                        rotation_vectors, translation_vectors,
                        camera_matrix, distortion_coefficients,
                        ret, allCorners, allIds,
                        camera=name, threshold=threshold)
                print(f"Each filtering {time.time()-t0}")

                # Apply guard again after bypass returns potentially thin images
                filtered_corners, filtered_ids, _ = _drop_thin_images(
                    filtered_corners, filtered_ids,
                    label=f"{name}[loop iter {index}]")
                if not filtered_corners:
                    break

                t0 = time.time()
                try:
                    (ret, camera_matrix, distortion_coefficients,
                     rotation_vectors, translation_vectors,
                     _, _, perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                        charucoCorners=filtered_corners,
                        charucoIds=filtered_ids,
                        board=self.board,
                        imageSize=imsize,
                        cameraMatrix=cameraMatrixInit,
                        distCoeffs=distCoeffsInit,
                        flags=flags,
                        criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 50000, 1e-9))
                except Exception as e:
                    print(f"[{name}] calibrateCameraCharucoExtended loop failed: {e}")
                    return (ret, camera_matrix, distortion_coefficients,
                            rotation_vectors, translation_vectors,
                            filtered_ids, filtered_corners, allCorners, allIds)

                cameraMatrixInit        = camera_matrix
                distCoeffsInit          = distortion_coefficients
                threshold               = 5 * imsize[1]/800.
                print(f"Each calibration {time.time()-t0}")
                index += 1
                if (index > 5 or (previous_ids == [] + filtered_ids
                                  and len(previous_ids) >= len(filtered_ids)
                                  and index > 2)):
                    print(f"Whole procedure: {time.time()-whole}")
                    break
                previous_ids = filtered_ids
        except Exception as e:
            return (f"Failed to calibrate camera {name}: {e}",
                    None, None, None, None, None, None, None, None)

        if self.traceLevel in (3, 10):
            print('Per View Errors...\n', perViewErrors)
        return (ret, camera_matrix, distortion_coefficients,
                rotation_vectors, translation_vectors,
                filtered_ids, filtered_corners, allCorners, allIds)

    # ──────────────────────────────────────────────────────────────────────────
    def calibrate_fisheye(self, allCorners, allIds, imsize, hfov, name):
        obj_points = [self.charuco_ids_to_objpoints(ids) for ids in allIds]
        f_init = imsize[0] / np.deg2rad(hfov) * 1.15
        Ki = np.array([[f_init,0.,imsize[0]/2],[0.,f_init,imsize[1]/2],[0.,0.,1.]])
        Di = np.zeros((4,1))
        rvecs, tvecs = [], []
        self.index = 0
        for corners, ids in zip(allCorners, allIds):
            objpts   = self.charuco_ids_to_objpoints(ids)
            cu       = cv2.fisheye.undistortPoints(corners, Ki, Di, None, np.eye(3))
            rv, tv, _= self.camera_pose_charuco(objpts, cu, ids, np.eye(3), np.array((0.,0,0,0)))
            tvecs.append(tv); rvecs.append(rv); self.index += 1

        cr, fi, fc = self.filter_corner_outliers(allIds, allCorners, Ki, Di, rvecs, tvecs)
        if cr:
            obj_points = [self.charuco_ids_to_objpoints(ids) for ids in fi]

        flags = (cv2.fisheye.CALIB_USE_INTRINSIC_GUESS + cv2.fisheye.CALIB_CHECK_COND
                 + cv2.fisheye.CALIB_FIX_SKEW + cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                 + self.get_fisheye_distortion_flags(name))
        tc = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 30, 1e-9)
        try:
            res, K, d, rv, tv = cv2.fisheye.calibrate(obj_points, fc, None, Ki, Di,
                                                       flags=flags, criteria=tc)
        except:
            success = False; crop = 0.95
            while not success:
                ol, cl = [], []
                for op, corners in zip(obj_points, fc):
                    ot, ct = [], []
                    for p, c in zip(op, corners):
                        if (imsize[0]*(1-crop) < c[0,0] < imsize[0]*crop
                                and imsize[1]*(1-crop) < c[0,1] < imsize[1]*crop):
                            ot.append(p); ct.append(c)
                    ol.append(np.array(ot)); cl.append(np.array(ct))
                try:
                    res, K, d, rv, tv = cv2.fisheye.calibrate(ol, cl, None, Ki, Di,
                                                               flags=flags, criteria=tc)
                    success = True
                except:
                    if crop > 0.7: crop -= 0.05
                    else: raise Exception("Fisheye calibration failed at max crop")
            try: res, K, d, rv, tv = cv2.fisheye.calibrate(obj_points, fc, imsize, K, Di,
                                                             flags=flags, criteria=tc)
            except: pass

        fc2, fi2, _, _, _, _ = self.features_filtering_function(
            rv, tv, K, d, res, allCorners, allIds, camera=name, threshold=1)
        return res, K, d, rv, tv, fi2, fc2

    # ──────────────────────────────────────────────────────────────────────────
    def calibrate_stereo(self, left_name, right_name,
                         allIds_l, allCorners_l, allIds_r, allCorners_r,
                         cameraMatrix_l, distCoeff_l, cameraMatrix_r, distCoeff_r,
                         t_in, r_in, features=None):

        one_pts = (self.board.chessboardCorners if self.board is not None
                   else self._plain_objpoints())

        # pose + filtering for each side
        for side_name, sCorners, sIds, sK, sD in [
                (left_name,  allCorners_l, allIds_l, cameraMatrix_l, distCoeff_l),
                (right_name, allCorners_r, allIds_r, cameraMatrix_r, distCoeff_r)]:
            rvecs, tvecs = [], []
            for corners, ids in zip(sCorners, sIds):
                objpts = self.charuco_ids_to_objpoints(ids)
                rv, tv, _ = self.camera_pose_charuco(objpts, corners, ids, sK, sD)
                rvecs.append(rv); tvecs.append(tv)
            if side_name == left_name:
                allCorners_l, allIds_l, _, _, _, _ = self.features_filtering_function(
                    rvecs, tvecs, sK, sD, 0., sCorners, sIds, camera=side_name, threshold=1)
            else:
                allCorners_r, allIds_r, _, _, _, _ = self.features_filtering_function(
                    rvecs, tvecs, sK, sD, 0., sCorners, sIds, camera=side_name, threshold=1)

        if self.traceLevel in (2, 4, 10):
            print(f'allIds_l: {len(allIds_l)}  allIds_r: {len(allIds_r)}')

        # find common corners across stereo pairs
        left_corners_sampled, right_corners_sampled, obj_pts = [], [], []
        for i in range(min(len(allIds_l), len(allIds_r))):
            lsc, rsc, ops = [], [], []
            for j in range(len(allIds_l[i])):
                idx = np.where(allIds_r[i] == allIds_l[i][j])[0]
                if len(idx) == 0: continue
                lsc.append(allCorners_l[i][j])
                rsc.append(allCorners_r[i][idx[0]])
                ops.append(one_pts[allIds_l[i][j]])
            if len(lsc) < 20:
                print(f"Skipping pair {i}: only {len(lsc)} common points (<20)"); continue
            print(f"ACCEPTED pair {i}: {len(lsc)} common points")
            obj_pts.append(np.array(ops, dtype=np.float32))
            left_corners_sampled.append(np.array(lsc, dtype=np.float32))
            right_corners_sampled.append(np.array(rsc, dtype=np.float32))

        if len(obj_pts) == 0:
            return -1, "Stereo calibration failed — no pairs with ≥5 common points"

        sc  = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 300, 1e-9)
        flags = (cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_USE_EXTRINSIC_GUESS
                 | cv2.CALIB_FIX_S1_S2_S3_S4 | cv2.CALIB_RATIONAL_MODEL
                 | self.get_distortion_flags(left_name))

        print("[Stereo] Applying robust dtype/shape fix...")
        for i in range(len(obj_pts)):
            obj_pts[i]              = np.asarray(obj_pts[i],              dtype=np.float32).reshape(-1,1,3)
            left_corners_sampled[i] = np.asarray(left_corners_sampled[i], dtype=np.float32).reshape(-1,1,2)
            right_corners_sampled[i]= np.asarray(right_corners_sampled[i],dtype=np.float32).reshape(-1,1,2)

        cL  = np.asarray(cameraMatrix_l, dtype=np.float64)
        cR  = np.asarray(cameraMatrix_r, dtype=np.float64)
        dL  = np.asarray(distCoeff_l,    dtype=np.float64).ravel()
        dR  = np.asarray(distCoeff_r,    dtype=np.float64).ravel()
        Rin = np.asarray(r_in, dtype=np.float64)
        Tin = np.asarray(t_in, dtype=np.float64)
        imageSize = (self.width.get(left_name, 2328), self.height.get(left_name, 1748))
        print(f"[Stereo] imageSize = {imageSize}")

        ret, M1, d1, M2, d2, R, T, E, F, pVE = cv2.stereoCalibrateExtended(
            obj_pts, left_corners_sampled, right_corners_sampled,
            cL, dL, cR, dR, imageSize, R=Rin, T=Tin, criteria=sc, flags=flags)

        # ══════════════════════════════════════════════════════════════════════
        # FIX — raised epipolar outlier threshold 5 → 20 px
        # At 640×360 the image diagonal is ~730 px; 5 px removes everything.
        # ══════════════════════════════════════════════════════════════════════
        epi_threshold = 8.0
        if pVE is not None and np.any(np.array(pVE).T[0] > epi_threshold):
            print(f"Removing pairs with epipolar error > {epi_threshold} px")
            bad = [i for i, e in enumerate(np.array(pVE).T[0]) if e > epi_threshold]
            for i in sorted(bad, reverse=True):
                del obj_pts[i]; del left_corners_sampled[i]; del right_corners_sampled[i]
            print(f"Removed {len(bad)} pairs — {len(obj_pts)} remaining")
            if len(obj_pts) == 0:
                return -1, "All stereo pairs removed by epipolar threshold"
            for i in range(len(obj_pts)):
                obj_pts[i]              = np.asarray(obj_pts[i],              dtype=np.float32).reshape(-1,1,3)
                left_corners_sampled[i] = np.asarray(left_corners_sampled[i], dtype=np.float32).reshape(-1,1,2)
                right_corners_sampled[i]= np.asarray(right_corners_sampled[i],dtype=np.float32).reshape(-1,1,2)
            ret, M1, d1, M2, d2, R, T, E, F, pVE = cv2.stereoCalibrateExtended(
                obj_pts, left_corners_sampled, right_corners_sampled,
                cL, dL, cR, dR, imageSize, R=Rin, T=Tin, criteria=sc, flags=flags)

        r_euler = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
        print(f'Epipolar error: {ret:.5f} px')
        print('R =\n', R, '\nT =\n', T)
        print(f'Euler (xyz deg): {r_euler}')

        R_l, R_r, P_l, P_r, Q, _, _ = cv2.stereoRectify(cL, dL, cR, dR, imageSize, R, T)
        return [ret, R, T, R_l, R_r, P_l, P_r]

    def _plain_objpoints(self):
        sq = (self.square_size_cm/100.) if self.square_size_cm else 0.03
        nx, ny = self.squaresX, self.squaresY
        pts = np.zeros((nx*ny, 3), np.float32)
        pts[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) * sq
        return pts

    # ──────────────────────────────────────────────────────────────────────────
    def display_rectification(self, image_data_pairs, images_corners_l, images_corners_r,
                              image_epipolar_color, isHorizontal):
        for idx, pair in enumerate(image_data_pairs):
            img_concat = (cv2.hconcat(pair) if isHorizontal else cv2.vconcat(pair))
            for lp, rp, cm in zip(images_corners_l[idx], images_corners_r[idx],
                                  image_epipolar_color[idx]):
                if isHorizontal:
                    cv2.line(img_concat, (int(lp[0][0]), int(lp[0][1])),
                             (int(rp[0][0])+pair[0].shape[1], int(rp[0][1])), colors[cm], 1)
                else:
                    cv2.line(img_concat, (int(lp[0][0]), int(lp[0][1])),
                             (int(rp[0][0]), int(rp[0][1])+pair[0].shape[0]), colors[cm], 1)
            cv2.imshow('Stereo Pair', cv2.resize(img_concat, (0,0), fx=0.8, fy=0.8))
            if cv2.waitKey(1) == 27: break
        cv2.destroyWindow('Stereo Pair')

    def scale_image(self, img, scaled_res):
        if img.shape[0] == scaled_res[0] and img.shape[1] == scaled_res[1]: return img
        sw = scaled_res[1] / img.shape[1]
        img = cv2.resize(img, (int(img.shape[1]*sw), int(img.shape[0]*sw)), cv2.INTER_CUBIC)
        if img.shape[0] < scaled_res[0]:
            raise RuntimeError(f"Resized height {img.shape[0]} < required {scaled_res[0]}")
        dh = (img.shape[0]-scaled_res[0])//2
        return img[dh:dh+scaled_res[0], :]

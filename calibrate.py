#!/usr/bin/env python3
"""
IMX519 Dual-Camera Stereo Calibration — Raspberry Pi 5
Works with existing datasets (640×360 or any resolution).
Outputs calibration JSON with cameras_1920x1080_@30fps block.

Usage (existing dataset):
  python3 calibrate.py -s 2.0 -ms 1.1 -nx 11 -ny 8 -m process -dst calib_images
"""

import cv2
import numpy as np
import time
import json
import math
import shutil
import threading
import traceback
import argparse
from pathlib import Path
from datetime import datetime

try:
    from picamera2 import Picamera2
    _PICAMERA2_AVAILABLE = True
except ImportError:
    _PICAMERA2_AVAILABLE = False

try:
    import calibration_utils as calibUtils
except ImportError:
    print("ERROR: calibration_utils.py not found in same directory.")
    raise

# ── Hardware constants ─────────────────────────────────────────────────────────
IMX519_WIDTH    = 2328
IMX519_HEIGHT   = 1748
IMX519_FPS      = 30.0
_FRAME_DUR_US   = int(1_000_000 / IMX519_FPS)
IMX519_HFOV_DEG = 66.0
BASELINE_CM     = 8.0

_GREEN = (0, 255,   0)
_RED   = (0,   0, 255)
_WHITE = (255, 255, 255)
_font  = cv2.FONT_HERSHEY_SIMPLEX


# ── Board config ──────────────────────────────────────────────────────────────
def make_board_config(baseline_cm=BASELINE_CM, hfov=IMX519_HFOV_DEG):
    return {
        "cameras": {
            "CAM_B": {
                "name": "left",
                "type": "color",
                "hfov": hfov,
                "extrinsics": {
                    "to_cam": "CAM_C",
                    "specTranslation": {"x": -baseline_cm, "y": 0.0, "z": 0.0},
                    "rotation": {"r": 0.0, "p": 0.0, "y": 0.0},
                },
            },
            "CAM_C": {
                "name": "right",
                "type": "color",
                "hfov": hfov,
            },
        },
        "stereo_config": {"left_cam": "CAM_B", "right_cam": "CAM_C"},
    }


# ── Argument parser ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="IMX519 stereo calibration. Use -m process -dst <folder> for existing images.",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    b = p.add_argument_group("Charuco board  (REQUIRED)")
    b.add_argument("-s",  "--squareSizeCm",  type=float, required=True)
    b.add_argument("-ms", "--markerSizeCm",  type=float, default=None)
    b.add_argument("-nx", "--squaresX",      type=int,   default=11)
    b.add_argument("-ny", "--squaresY",      type=int,   default=8)

    c = p.add_argument_group("Camera / rig")
    c.add_argument("--baseline-cm",   type=float, default=BASELINE_CM)
    c.add_argument("--hfov",          type=float, default=IMX519_HFOV_DEG)
    c.add_argument("--left-cam-idx",  type=int,   default=0)
    c.add_argument("--right-cam-idx", type=int,   default=1)
    c.add_argument("-cm", "--cameraMode", type=str, default="perspective",
                   choices=["perspective", "fisheye"])

    cap = p.add_argument_group("Capture")
    cap.add_argument("-m",    "--mode",    nargs="*", default=["capture","process"])
    cap.add_argument("-c",    "--count",   type=int,  default=3)
    cap.add_argument("-cd",   "--captureDelay", type=int, default=2)
    cap.add_argument("-ep",   "--maxEpipolarError", type=float, default=0.8)
    cap.add_argument("-mdmp", "--minDetectedMarkersPercent", type=float, default=0.4)
    cap.add_argument("-ebp",  "--enablePolygonsDisplay", action="store_true")
    cap.add_argument("-iv",   "--invertVertical",   action="store_true")
    cap.add_argument("-ih",   "--invertHorizontal", action="store_true")
    cap.add_argument("-rd",   "--rectifiedDisp", default=True, action="store_false")

    d = p.add_argument_group("Output")
    d.add_argument("-osf", "--outputScaleFactor", type=float, default=0.4)
    d.add_argument("-dst", "--datasetPath",  type=str, default="dataset")
    d.add_argument("-out", "--outputPath",   type=str, default="")
    d.add_argument("-trc", "--traceLevel",   type=int, default=0)

    args = p.parse_args()
    if args.markerSizeCm is None:
        args.markerSizeCm = round(args.squareSizeCm * 0.75, 4)
    if args.squareSizeCm < 1.0:   p.error("-s must be >= 1.0 cm")
    if args.markerSizeCm >= args.squareSizeCm: p.error("markerSize must be < squareSize")
    if args.baseline_cm  <= 0:    p.error("--baseline-cm must be positive")
    return args


# ── Dual camera (live capture only) ───────────────────────────────────────────
class DualIMX519:
    def __init__(self, left_idx=0, right_idx=1):
        if not _PICAMERA2_AVAILABLE:
            raise RuntimeError("picamera2 not available — use -m process for offline mode")
        self.cam_left  = Picamera2(left_idx)
        self.cam_right = Picamera2(right_idx)
        for cam in (self.cam_left, self.cam_right):
            cam.configure(cam.create_video_configuration(
                main={"size": (IMX519_WIDTH, IMX519_HEIGHT), "format": "BGR888"},
                controls={"FrameDurationLimits": (_FRAME_DUR_US, _FRAME_DUR_US),
                          "AeEnable": True, "AwbEnable": True,
                          "NoiseReductionMode": 0, "Sharpness": 1.0},
                buffer_count=4))
        self._running = False

    def start(self):
        self.cam_left.start(); self.cam_right.start()
        print("[Camera] Waiting 2 s for AE/AWB ..."); time.sleep(2.0)
        self._running = True

    def stop(self):
        if self._running:
            self.cam_left.stop(); self.cam_right.stop(); self._running = False
        self.cam_left.close(); self.cam_right.close()

    def get_preview(self):
        return self.cam_left.capture_array("main"), self.cam_right.capture_array("main")

    def capture_sync(self):
        frames  = [None, None]
        barrier = threading.Barrier(2)
        def _grab(cam, slot):
            barrier.wait(); frames[slot] = cam.capture_array("main")
        t0 = threading.Thread(target=_grab, args=(self.cam_left,  0), daemon=True)
        t1 = threading.Thread(target=_grab, args=(self.cam_right, 1), daemon=True)
        t0.start(); t1.start(); t0.join(); t1.join()
        return frames[0], frames[1]


# ── Main class ─────────────────────────────────────────────────────────────────
class IMX519StereoCalib:

    polygons                = None
    current_polygon         = 0
    images_captured_polygon = 0
    images_captured         = 0

    def __init__(self):
        self.args  = parse_args()
        self.trace = self.args.traceLevel
        self.scale = self.args.outputScaleFactor

        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
        if self.args.markerSizeCm > 0:
            print("Charuco mode activated (marker size > 0)")
            self.charuco_board = cv2.aruco.CharucoBoard_create(
                self.args.squaresX, self.args.squaresY,
                self.args.squareSizeCm, self.args.markerSizeCm, self.aruco_dict)
            self.is_charuco = True
        else:
            print("Plain chessboard mode")
            self.charuco_board = None
            self.is_charuco    = False

        self.board_config = make_board_config(
            baseline_cm=self.args.baseline_cm, hfov=self.args.hfov)
        _ref_polys        = calibUtils.setPolygonCoordinates(1000, 600)
        self.total_images = self.args.count * len(_ref_polys)
        self.coverage_images = {info["name"]: None
                                for info in self.board_config["cameras"].values()}
        self.dataset_path = self.args.datasetPath

        print(f"\n  Dataset  : {self.args.datasetPath}")
        print(f"  Board    : {self.args.squaresX}×{self.args.squaresY} Charuco  "
              f"sq={self.args.squareSizeCm}cm  mk={self.args.markerSizeCm}cm\n")

    # ── Charuco helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _gray(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    def _is_markers_found(self, frame):
        corners, _, _ = cv2.aruco.detectMarkers(self._gray(frame), self.aruco_dict)
        req = int(math.floor(self.args.squaresX * self.args.squaresY / 2)
                  * self.args.minDetectedMarkersPercent)
        return len(corners) >= req

    def _detect_corners(self, frame):
        gray = self._gray(frame)
        mc, mi, rej = cv2.aruco.detectMarkers(gray, self.aruco_dict)
        mc, mi, _, _ = cv2.aruco.refineDetectedMarkers(
            gray, self.charuco_board, mc, mi, rejectedCorners=rej)
        if len(mc) == 0: return mc, mi, None, None
        _, cc, ci = cv2.aruco.interpolateCornersCharuco(
            mc, mi, gray, self.charuco_board, minMarkers=1)
        return mc, mi, cc, ci

    def _draw_markers(self, frame):
        out = frame.copy()
        _, _, cc, ci = self._detect_corners(out)
        if ci is not None and len(ci) > 0:
            cv2.aruco.drawDetectedCornersCharuco(out, cc, ci, _GREEN)
        return out

    def _draw_on_coverage(self, src, canvas, color):
        _, _, cc, _ = self._detect_corners(src)
        if cc is not None:
            r = max(4, 8 * canvas.shape[1] // 1920)
            for corner in cc:
                cv2.circle(canvas, (int(corner[0][0]), int(corner[0][1])), r, color, -1)
        return canvas

    # ── UI helpers ─────────────────────────────────────────────────────────────
    def _show_info_frame(self):
        h, w = 600, 1000
        info = np.zeros((h, w, 3), np.uint8); info[:70] = (0, 45, 0)
        def put(y, t, s=0.8, c=_GREEN, th=2):
            cv2.putText(info, t, (30, y), _font, s, c, th)
        put(48,  "IMX519 Stereo Calibration  —  Raspberry Pi 5", 1.0, _WHITE, 2)
        put(110, f"Board : {self.args.squaresX}×{self.args.squaresY} Charuco  "
                 f"sq={self.args.squareSizeCm}cm  mk={self.args.markerSizeCm}cm")
        put(150, "SPACE → capture    S → stop early    ESC/Q → abort")
        cv2.imshow("IMX519 Calibration", info)
        while True:
            k = cv2.waitKey(10) & 0xFF
            if k == ord(" "): cv2.destroyAllWindows(); return
            if k in (27, ord("q")): cv2.destroyAllWindows(); raise SystemExit(0)

    def _show_failed(self, reason):
        img = np.zeros((200, 700, 3), np.uint8); img[:] = (20,0,0)
        cv2.putText(img, "CAPTURE FAILED", (30, 80), _font, 1.2, _RED, 3)
        cv2.putText(img, reason,           (30,130), _font, 0.7, _WHITE, 2)
        cv2.imshow("IMX519 Calibration", img); cv2.waitKey(1400)

    def _get_rotated_polygon(self, polygon, dh, dw):
        local = np.array([polygon])
        if self.images_captured_polygon == 0: return local
        angle = 30. if self.images_captured_polygon == 1 else -30.
        theta = np.deg2rad(angle)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rotated = np.matmul(local, R).astype(np.int32)
        if self.images_captured_polygon == 1:
            rotated[0][:,1] += abs(rotated.min())
        else:
            rotated[0][:,1] += dh - abs(rotated[0][:,1].max())
            rotated[0][:,0] += abs(rotated[0][:,1].min())
        return rotated

    def _save_frame(self, frame, cam_name):
        if not self._is_markers_found(frame): return False
        fname = calibUtils.image_filename(self.current_polygon, self.images_captured)
        path  = Path(self.dataset_path) / cam_name / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        return cv2.imwrite(str(path), frame)

    # ── Live capture ───────────────────────────────────────────────────────────
    def capture_images(self):
        WIN = "IMX519 Calibration"
        finished = capturing = start_timer = False
        timer = self.args.captureDelay; prev_time = 0.

        cam = DualIMX519(left_idx=self.args.left_cam_idx, right_idx=self.args.right_cam_idx)
        cam.start()
        try:
            while not finished:
                lr, rr = cam.get_preview()
                if self.args.invertVertical and self.args.invertHorizontal:
                    lr = cv2.flip(lr,-1); rr = cv2.flip(rr,-1)
                elif self.args.invertVertical:
                    lr = cv2.flip(lr,0); rr = cv2.flip(rr,0)
                elif self.args.invertHorizontal:
                    lr = cv2.flip(lr,1); rr = cv2.flip(rr,1)

                ld = cv2.resize(self._draw_markers(lr), (0,0), fx=self.scale, fy=self.scale)
                rd = cv2.resize(self._draw_markers(rr), (0,0), fx=self.scale, fy=self.scale)
                dh, dw = ld.shape[:2]

                if self.polygons is None:
                    self.polygons = calibUtils.setPolygonCoordinates(dh, dw)

                if self.args.enablePolygonsDisplay and self.current_polygon < len(self.polygons):
                    poly = self._get_rotated_polygon(self.polygons[self.current_polygon], dh, dw)
                    cv2.polylines(ld, poly, True, _RED, 3)
                    cv2.polylines(rd, poly, True, _RED, 3)

                cv2.putText(ld, f"LEFT  {self.images_captured}/{self.total_images}",
                            (10,30), _font, 0.8, _GREEN, 2)
                cv2.putText(rd, f"RIGHT pos {self.current_polygon+1}/{len(self.polygons)}",
                            (10,30), _font, 0.8, _GREEN, 2)
                combined = np.hstack([ld, rd])

                if start_timer:
                    now = time.time()
                    if now - prev_time >= 1.: timer -= 1; prev_time = now
                    if timer <= 0: start_timer = False; capturing = True
                    else:
                        ch, cw = combined.shape[:2]
                        cv2.putText(combined, str(timer), (cw//2-50, ch//2+60),
                                    _font, 9, _RED, 6, cv2.LINE_AA)

                cv2.imshow(WIN, combined)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")): raise SystemExit(0)
                elif key == ord(" "): start_timer=True; prev_time=time.time(); timer=self.args.captureDelay
                elif key == ord("s"): finished=True; break

                if capturing:
                    capturing = False
                    lc, rc = cam.capture_sync()
                    okl = self._save_frame(lc, "left")
                    okr = self._save_frame(rc, "right")
                    if okl and okr:
                        col = tuple(int(x) for x in np.random.randint(40,230,3))
                        for nm, frm in [("left",lc),("right",rc)]:
                            if self.coverage_images[nm] is None:
                                self.coverage_images[nm] = np.ones_like(frm)*255
                            self.coverage_images[nm] = self._draw_on_coverage(
                                frm, self.coverage_images[nm], col)
                        self.images_captured += 1; self.images_captured_polygon += 1
                    else:
                        for nm in ("left","right"):
                            bad = (Path(self.dataset_path)/nm/
                                   calibUtils.image_filename(self.current_polygon, self.images_captured))
                            bad.unlink(missing_ok=True)
                        self._show_failed("LEFT" if not okl else "RIGHT" if not okr
                                          else "Both" + " markers missing")

                    if self.images_captured_polygon >= self.args.count:
                        self.images_captured_polygon = 0; self.current_polygon += 1
                        if self.current_polygon >= len(self.polygons): finished = True
        finally:
            cam.stop(); cv2.destroyAllWindows()

    # ── Calibration ────────────────────────────────────────────────────────────
    def calibrate(self):
        print("\n" + "="*65)
        print("  STEREO CALIBRATION  (calibration_utils.StereoCalibration)")
        print("="*65)

        for name in ("left", "right"):
            imgs = list((Path(self.dataset_path)/name).glob("*.png"))
            if not imgs:
                print(f"  ERROR: No images in {self.dataset_path}/{name}/"); raise SystemExit(1)
            print(f"  {name}: {len(imgs)} images")

        stereo_calib = calibUtils.StereoCalibration(
            traceLevel=self.args.traceLevel,
            outputScaleFactor=self.args.outputScaleFactor,
            disableCamera=[],
            square_size=self.args.squareSizeCm)

        try:
            status, result_config = stereo_calib.calibrate(
                board_config=self.board_config,
                filepath=self.dataset_path,
                square_size=self.args.squareSizeCm,
                mrk_size=self.args.markerSizeCm,
                squaresX=self.args.squaresX,
                squaresY=self.args.squaresY,
                camera_model=self.args.cameraMode,
                enable_disp_rectify=self.args.rectifiedDisp,
                intrinsic_img={}, extrinsic_img={}, charucos={})
        except calibUtils.StereoExceptions as exc:
            print(f"\n  CALIBRATION ERROR: {exc.summary}"); raise SystemExit(1)
        except Exception:
            print("\n  CALIBRATION ERROR (unexpected):"); traceback.print_exc(); raise SystemExit(1)

        if status != 1:
            print(f"\n  FAILED  status={status}"); raise SystemExit(1)

        # ── build output JSON ──────────────────────────────────────────────────
        errors = []
        output = {
            "calibration_date"    : datetime.now().isoformat(),
            "sensor"              : "IMX519",
            "sensor_mode"         : f"{IMX519_WIDTH}x{IMX519_HEIGHT}@{IMX519_FPS}fps",
            "hfov_used_deg"       : self.args.hfov,
            "hfov_note"           : ("HORIZONTAL FOV. pixel_pitch=1.22um, "
                                     "diag_FOV=77.7deg → HFOV=65.6deg ≈ 66deg"),
            "camera_model"        : self.args.cameraMode,
            "baseline_cm_spec"    : self.args.baseline_cm,
            "charuco_squareSizeCm": self.args.squareSizeCm,
            "charuco_markerSizeCm": self.args.markerSizeCm,
            "charuco_squaresX"    : self.args.squaresX,
            "charuco_squaresY"    : self.args.squaresY,
            "cameras"             : {},
            "stereo"              : {},
        }

        print("\n  Results:\n  " + "-"*55)
        cal_w = cal_h = None

        for cam_key, cam_info in result_config["cameras"].items():
            name = cam_info["name"]
            if "reprojection_error" not in cam_info: continue

            rep_err = cam_info["reprojection_error"]
            rep_thr = max(1.0, cam_info["size"][1] / 720.0)
            tag     = "OK" if rep_err <= rep_thr else "HIGH"
            if rep_err > rep_thr:
                errors.append(f"High reprojection on '{name}': {rep_err:.4f} px")
            print(f"  [{name:5s}]  reprojection = {rep_err:.5f} px  [{tag}]")

            if cal_w is None: cal_w, cal_h = cam_info["size"]

            output["cameras"][name] = {
                "intrinsics"        : cam_info["intrinsics"].tolist(),
                "dist_coeff"        : cam_info["dist_coeff"].tolist(),
                "image_size_wh"     : list(cam_info["size"]),
                "reprojection_error": rep_err,
            }

            if "extrinsics" in cam_info and "to_cam" in cam_info["extrinsics"]:
                epi    = cam_info["extrinsics"].get("epipolar_error", -1)
                epi_ok = 0 <= epi <= self.args.maxEpipolarError
                if not epi_ok:
                    errors.append(f"High epipolar: {epi:.5f} px (limit {self.args.maxEpipolarError})")
                print(f"  [stereo]   epipolar  = {epi:.5f} px  "
                      f"[{'OK' if epi_ok else 'HIGH'}]")
                R  = cam_info["extrinsics"]["rotation_matrix"]
                T  = cam_info["extrinsics"]["translation"]
                bl = float(np.linalg.norm(T))
                print(f"  [stereo]   baseline  = {bl:.4f} cm")
                output["stereo"].update({
                    "epipolar_error"    : epi,
                    "rotation_matrix"   : R.tolist(),
                    "translation_cm"    : T.tolist(),
                    "baseline_cm_solved": bl,
                })

        sc = result_config.get("stereo_config", {})
        if "rectification_left" in sc:
            output["stereo"]["rectification_left"]  = sc["rectification_left"].tolist()
            output["stereo"]["rectification_right"] = sc["rectification_right"].tolist()
            print("  Rectification matrices saved.")

        # ── scale to deployment resolutions ────────────────────────────────────
        # Rule:  fx_new = fx_cal × (W_new / W_cal)  etc.
        # Valid for same-aspect-ratio rescales (no crop, same FOV).
        # dist_coeff, R, T, rectification are all pixel-independent — unchanged.
        if cal_w is not None:
            print(f"\n  Calibration resolution: {cal_w}×{cal_h}")
            for label, (tw, th) in {
                    "1920x1080_@30fps": (1920, 1080),
                    "1280x720_@60fps":  (1280,  720),
                    "640x360_@90fps":   ( 640,  360)}.items():
                sx = tw / cal_w
                sy = th / cal_h
                if abs(sx - sy) > 0.01:
                    print(f"  [SKIP] {label}: aspect mismatch — needs separate calibration")
                    continue
                block = {}
                for cam_name, cam_data in output["cameras"].items():
                    K  = np.array(cam_data["intrinsics"])
                    Ks = K.copy()
                    Ks[0,0] *= sx; Ks[1,1] *= sx   # fx, fy
                    Ks[0,2] *= sx; Ks[1,2] *= sx   # cx, cy
                    block[cam_name] = {
                        "intrinsics"        : Ks.tolist(),
                        "dist_coeff"        : cam_data["dist_coeff"],
                        "image_size_wh"     : [tw, th],
                        "reprojection_error": cam_data["reprojection_error"],
                        "scale_factor"      : round(sx, 6),
                        "scaled_from"       : f"{cal_w}x{cal_h}",
                        "stereo_note"       : ("Use stereo.rotation_matrix, translation_cm, "
                                              "rectification_left/right unchanged from top-level.")
                    }
                block_key = "cameras_" + label
                output[block_key] = block
                if label == "1920x1080_@30fps":
                    for cn, cd in block.items():
                        if cn == "stereo_note": continue
                        K = cd["intrinsics"]
                        print(f"  [{cn} @1080p] fx={K[0][0]:.2f}  fy={K[1][1]:.2f}  "
                              f"cx={K[0][2]:.2f}  cy={K[1][2]:.2f}")
                print(f"  Saved: cameras_{label}  (scale ×{sx:.4g})")

        # ── write files ────────────────────────────────────────────────────────
        out_path = (Path(self.args.outputPath) if self.args.outputPath
                    else Path(f"imx519_stereo_calib_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  JSON saved  →  {out_path.resolve()}")

        with open(Path(self.dataset_path)/"target_info.txt", "w") as f:
            f.write(f"Date              : {output['calibration_date']}\n")
            f.write(f"HFOV used         : {self.args.hfov} deg (HORIZONTAL)\n")
            f.write(f"Square Size       : {self.args.squareSizeCm} cm\n")
            f.write(f"Marker Size       : {self.args.markerSizeCm} cm\n")
            f.write(f"Baseline (spec)   : {self.args.baseline_cm} cm\n")
            if "baseline_cm_solved" in output["stereo"]:
                f.write(f"Baseline (solved) : {output['stereo']['baseline_cm_solved']:.4f} cm\n")
            for nm, info in output["cameras"].items():
                f.write(f"\n[{nm}]  reprojection={info['reprojection_error']:.6f}  "
                        f"size={info['image_size_wh']}\n")
            if "epipolar_error" in output["stereo"]:
                f.write(f"\nepipolar_error: {output['stereo']['epipolar_error']:.6f}\n")
            b1080 = output.get("cameras_1920x1080_@30fps", {})
            if b1080:
                f.write("\n[1920x1080 intrinsics]\n")
                for nm in ("left","right"):
                    if nm not in b1080: continue
                    K = b1080[nm]["intrinsics"]
                    f.write(f"  [{nm}] fx={K[0][0]:.3f} fy={K[1][1]:.3f} "
                            f"cx={K[0][2]:.3f} cy={K[1][2]:.3f}\n")

        # ── result window ──────────────────────────────────────────────────────
        print()
        if errors:
            for e in errors: print(f"  WARNING: {e}")
            img = np.zeros((300, 1000, 3), np.uint8); img[:] = (20,0,40)
            cv2.putText(img, "Done — check warnings", (30,60), _font, 1.0, _RED, 2)
            for i, e in enumerate(errors[:4]):
                cv2.putText(img, e, (30, 120+i*50), _font, 0.65, (200,180,255), 2)
        else:
            print("  Calibration SUCCESSFUL.")
            img = np.zeros((320, 1000, 3), np.uint8); img[:] = (0,40,0)
            cv2.putText(img, "Calibration SUCCESSFUL", (30,65), _font, 1.8, _GREEN, 3)
            if "baseline_cm_solved" in output["stereo"]:
                b = output["stereo"]["baseline_cm_solved"]
                cv2.putText(img, f"Baseline: {b:.4f} cm  (spec: {self.args.baseline_cm} cm)",
                            (30,135), _font, 0.80, _WHITE, 2)
            if "epipolar_error" in output["stereo"]:
                e = output["stereo"]["epipolar_error"]
                cv2.putText(img, f"Epipolar: {e:.5f} px  (limit: {self.args.maxEpipolarError} px)",
                            (30,185), _font, 0.80, _WHITE, 2)
            b1080 = output.get("cameras_1920x1080_@30fps", {})
            if "left" in b1080:
                K = b1080["left"]["intrinsics"]
                cv2.putText(img, f"1080p left: fx={K[0][0]:.1f}  cx={K[0][2]:.1f}",
                            (30,240), _font, 0.75, (180,255,180), 2)
            cv2.putText(img, str(out_path.name), (30,290), _font, 0.65, (200,255,200), 2)

        cv2.imshow("Calibration Result", img)
        cv2.waitKey(0); cv2.destroyAllWindows()

    # ── Entry point ────────────────────────────────────────────────────────────
    def run(self):
        if "capture" in self.args.mode:
            ds = Path(self.dataset_path)
            if ds.exists(): shutil.rmtree(ds)
            for info in self.board_config["cameras"].values():
                (ds/info["name"]).mkdir(parents=True, exist_ok=True)
            self._show_info_frame()
            self.capture_images()
        if "process" in self.args.mode:
            self.calibrate()
        print("\n[Done]")


if __name__ == "__main__":
    IMX519StereoCalib().run()

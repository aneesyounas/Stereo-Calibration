# IMX519 Stereo System — Raspberry Pi 5

This repository contains a **minimal stereo pipeline** for dual IMX519 cameras:

1. Stereo preview (camera bring-up & sync check)
2. Stereo calibration (Charuco-based)
3. ArUco stereo measurement (mm-accurate size & distance)

---

## 1. Stereo Preview (Camera Bring-Up)

### File

`IMX_519_preview.py`

### What it does

* Initializes **two IMX519 cameras** with identical controls
* Captures low-resolution frames (640×360)
* Displays **side-by-side (Left | Right)** preview
* Shows real-time FPS

### Dependencies

```
numpy==1.26.4
opencv-contrib-python==4.5.5.64
picamera2==0.3.33
```

System: libcamera

### Run

```bash
python3 IMX_519_preview.py
```

Press `q` to quit.

### Use

* Verify both cameras are detected
* Check exposure/white balance consistency
* Confirm overlap & alignment before calibration

---

## 2. Stereo Calibration (Charuco)

### Overview

Generates stereo intrinsics, extrinsics, and rectification for IMX519 pair.

### Dependencies (exact versions)

```
numpy==1.26.4
opencv-contrib-python==4.5.5.64
scipy==1.17.1
matplotlib==3.10.8
picamera2==0.3.33
```

Install:

```bash
sudo pip3 install --break-system-packages --ignore-installed "numpy==1.26.4"
sudo pip3 install --break-system-packages \
  "opencv-contrib-python==4.5.5.64" \
  "scipy==1.17.1" \
  "matplotlib==3.10.8" \
  "picamera2==0.3.33"
```

> Note: OpenCV 4.5.5 requires NumPy 1.x. Do not install `opencv-python`.

### Existing dataset

```
dst/
  left/*.png
  right/*.png
```

```bash
python3 calibrate.py -s -ms -nx -ny  \
  -m process -dst 
```

### Live capture + process

```bash
vcgencmd get_camera   # should show detected=2
```

```bash
python3 calibrate.py -s -ms -nx -ny \
  -m capture process -dst  
```

Controls: `SPACE` capture, `S` stop, `ESC/Q` abort

### Parameters

```
-s              square size (cm)
-ms             marker size (cm)
-nx / -ny       board size (default 11x8)
-c              captures/region
-cd             countdown
-ep             max epipolar error (px)
--baseline-cm   physical baseline
```

### Output

```
imx519_stereo_calib_YYYYMMDD_HHMMSS.json
dst/target_info.txt
dst/left_coverage.png
```

---

## 3. ArUco Stereo Measurement

### File

`aruco_measurement.py`

### What it does

* Loads calibration JSON
* Rectifies both cameras (1280×720)
* Detects AprilTag/ArUco markers
* Triangulates corners → computes:

  * marker size (mm)
  * distance (m)
* Reports reprojection error (runtime quality)

### Dependencies

```
numpy==1.26.4
opencv-contrib-python==4.5.5.64
picamera2==0.3.33
```

### Run

```bash
python3 imx519_aruco_stereo.py --calib imx519_stereo_calib_*.json
```

### Self-measure (recommended)

```bash
python3 aruco_measurement.py --selfmeasure
```

* Compare on-screen size with ruler
* Update `MARKER_SIZES_MM`

### Options

```
--fast           disable subpixel refinement
--brightness     camera brightness
```

### Use

* Validate calibration accuracy (mm-level)
* Measure object size & distance
* Debug stereo geometry
* Correct printed marker scale errors

### Notes

* Best range: **40–80 cm**
* Keep markers centered
* Ensure correct print scale (disable “fit to page”)

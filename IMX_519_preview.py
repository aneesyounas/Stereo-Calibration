
import cv2
import time
import os
import numpy as np
from datetime import datetime
from picamera2 import Picamera2
from libcamera import controls

# ===============================
# Initialize BOTH Cameras
# ===============================
picamL = Picamera2(0)   # Left camera
picamR = Picamera2(1)   # Right camera

# ===============================
# Video Configuration (same for both)
# ===============================
config = picamL.create_video_configuration(
    main={"size": (1920, 1080), "format": "XRGB8888"},
    lores={"size": (640, 360), "format": "XRGB8888"},
    controls={
        "FrameRate": 8,
        "AeFlickerMode": controls.AeFlickerModeEnum.Auto,
    }
)

picamL.configure(config)
picamR.configure(config)

# ===============================
# Shared Controls (KEEP IDENTICAL!)
# ===============================
common_controls = {
    "AwbEnable": True,      # Auto white balance
    "AeEnable": True,       # Auto exposure
    "ExposureValue": 0.0,   # No exposure compensation
    "AeMeteringMode": controls.AeMeteringModeEnum.Matrix,

    "Brightness": 0.0,      # Neutral brightness
    "Contrast": 1.0,        # Natural contrast
    "Saturation": 1.0,      # Natural color
    "Sharpness": 1.0,       # Mild sharpening only

    "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.Fast,

    "LensPosition": 1.0     # Depends on focus distance
}

picamL.set_controls(common_controls)
picamR.set_controls(common_controls)

picamL.start()
picamR.start()

print("[INFO] Stereo cameras started")

# ===============================
# Main Loop
# ===============================
last_time = time.time()

try:
    while True:
        frameL = picamL.capture_array("lores")
        frameR = picamR.capture_array("lores")

        bgrL = cv2.cvtColor(frameL, cv2.COLOR_RGBA2BGR)
        bgrR = cv2.cvtColor(frameR, cv2.COLOR_RGBA2BGR)

        # FPS
        now = time.time()
        fps = 1 / (now - last_time)
        last_time = now

        cv2.putText(bgrL, "LEFT", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(bgrR, "RIGHT", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        stereo = np.hstack((bgrL, bgrR))
        cv2.putText(stereo, f"FPS: {fps:.1f}", (10, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    

        cv2.imshow("Stereo Preview (L | R)", stereo)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picamL.stop()
    picamR.stop()
    cv2.destroyAllWindows()


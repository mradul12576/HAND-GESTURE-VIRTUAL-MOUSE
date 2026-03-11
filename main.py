import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import os
import requests
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

pyautogui.FAILSAFE = False

# --- CONFIGURATION ---
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_tracker.task")
CAM_WIDTH, CAM_HEIGHT = 640, 480
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

SMOOTH_FACTOR_SLOW = 7
SMOOTH_FACTOR_FAST = 2
SPEED_THRESHOLD = 15

CLICK_THRESHOLD = 0.045
SCROLL_THRESHOLD = 0.04
DEBOUNCE_TIME = 0.3

CLR_ACCENT = (255, 0, 255)
CLR_SUCCESS = (0, 255, 127)
CLR_BG = (30, 30, 30)
CLR_TEXT = (240, 240, 240)

# --- DOWNLOAD MODEL ---
if not os.path.exists(MODEL_PATH):
    print("Downloading hand tracking model...")
    response = requests.get(MODEL_URL)
    response.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

# --- GLOBAL STATE ---
latest_result = None


def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result


# --- INITIALIZE HAND TRACKER ---
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=result_callback,
)
landmarker = vision.HandLandmarker.create_from_options(options)


def get_dist(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def draw_hud(img, action, fps):
    cv2.rectangle(img, (10, 10), (260, 110), CLR_BG, -1)
    cv2.rectangle(img, (10, 10), (260, 110), (100, 100, 100), 1)

    cv2.putText(img, "VIRTUAL MOUSE PRO", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.6, CLR_ACCENT, 1)
    cv2.line(img, (20, 45), (250, 45), (60, 60, 60), 1)

    cv2.putText(img, f"STATUS: {action}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_TEXT, 1)
    cv2.putText(img, f"ENGINE: {int(fps)} FPS", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

prev_x, prev_y = 0, 0
last_click_time = 0
fps_time = time.time()
last_action = "IDLE"
trail_points = []

print("VIRTUAL MOUSE PRO: RUNNING")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    landmarker.detect_async(mp_image, int(time.time() * 1000))

    if latest_result and latest_result.hand_landmarks:
        for landmarks in latest_result.hand_landmarks:
            thumb = landmarks[4]
            index = landmarks[8]
            middle = landmarks[12]
            ring = landmarks[16]
            pinky = landmarks[20]

            ix = np.interp(index.x, (0.2, 0.8), (0, SCREEN_WIDTH))
            iy = np.interp(index.y, (0.2, 0.8), (0, SCREEN_HEIGHT))

            move_dist = np.sqrt((ix - prev_x) ** 2 + (iy - prev_y) ** 2)
            smoothing = np.interp(
                move_dist,
                (0, SPEED_THRESHOLD),
                (SMOOTH_FACTOR_SLOW, SMOOTH_FACTOR_FAST),
            )

            curr_x = prev_x + (ix - prev_x) / smoothing
            curr_y = prev_y + (iy - prev_y) / smoothing

            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y
            last_action = "MOVING"

            cx, cy = int(index.x * CAM_WIDTH), int(index.y * CAM_HEIGHT)
            trail_points.append((cx, cy))
            if len(trail_points) > 10:
                trail_points.pop(0)

            for i in range(1, len(trail_points)):
                thickness = int(np.sqrt(10 / float(i + 1)) * 2)
                cv2.line(img, trail_points[i - 1], trail_points[i], CLR_ACCENT, thickness)

            if get_dist(index, thumb) < CLICK_THRESHOLD:
                if time.time() - last_click_time > DEBOUNCE_TIME:
                    pyautogui.click()
                    last_click_time = time.time()
                    last_action = "L-CLICK"
                cv2.circle(img, (cx, cy), 25, CLR_SUCCESS, 2)
                cv2.circle(img, (cx, cy), 15, CLR_SUCCESS, -1)

            elif get_dist(middle, thumb) < CLICK_THRESHOLD:
                if time.time() - last_click_time > DEBOUNCE_TIME:
                    pyautogui.rightClick()
                    last_click_time = time.time()
                    last_action = "R-CLICK"
                cv2.circle(
                    img,
                    (int(middle.x * CAM_WIDTH), int(middle.y * CAM_HEIGHT)),
                    25,
                    (255, 50, 50),
                    2,
                )

            elif get_dist(middle, ring) < SCROLL_THRESHOLD:
                scroll = 40 if middle.y < 0.5 else -40
                pyautogui.scroll(scroll)
                last_action = "SCROLLING"
                cv2.circle(
                    img,
                    (int(middle.x * CAM_WIDTH), int(middle.y * CAM_HEIGHT)),
                    20,
                    (255, 255, 0),
                    2,
                )

            elif get_dist(thumb, pinky) < 0.04 and get_dist(index, pinky) < 0.04:
                last_action = "EXITING"
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit

            else:
                cv2.circle(img, (cx, cy), 10, CLR_ACCENT, -1)
                cv2.circle(img, (cx, cy), 14, CLR_ACCENT, 1)

    fps = 1 / (time.time() - fps_time + 1e-6)
    fps_time = time.time()
    draw_hud(img, last_action, fps)

    x1, y1 = int(0.2 * CAM_WIDTH), int(0.2 * CAM_HEIGHT)
    x2, y2 = int(0.8 * CAM_WIDTH), int(0.8 * CAM_HEIGHT)
    cv2.rectangle(img, (x1, y1), (x2, y2), CLR_ACCENT, 1)

    cv2.imshow("Virtual Mouse Pro", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
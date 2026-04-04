import tkinter as tk
from PIL import Image, ImageTk
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

# --- MODEL SETUP ---
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = "hand_tracker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

latest_result = None
def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=result_callback,
)
landmarker = vision.HandLandmarker.create_from_options(options)

def get_dist(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# --- APP ---
class VirtualMouseApp:
    def __init__(self, root, bg_image_path):
        self.root = root
        self.root.title("Virtual Mouse Pro Dashboard")
        self.root.geometry("950x700")
        self.running = False
        self.cap = None
        self.last_click = 0

        # --- Set background image ---
        bg_img = Image.open(bg_image_path)
        bg_img = bg_img.resize((950, 700))
        self.bg_photo = ImageTk.PhotoImage(bg_img)
        self.bg_label = tk.Label(root, image=self.bg_photo)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # --- Foreground UI ---
        # Title
        tk.Label(root, text="🎮 Virtual Mouse Dashboard", font=("Arial", 20, "bold"),
                 fg="#00FFFF", bg="#000000").place(x=20, y=20)

        # Status
        self.status = tk.Label(root, text="Status: Idle", font=("Arial", 14),
                               fg="#00FF7F", bg="#000000")
        self.status.place(x=20, y=60)

        # Video frame
        self.video_label = tk.Label(root, bg="black")
        self.video_label.place(x=150, y=100, width=640, height=480)

        # Buttons frame
        self.btn_frame = tk.Frame(root, bg="#000000")
        self.btn_frame.place(x=150, y=600)

        # Buttons
        self.start_btn = self.create_button("▶ Start", "#4CAF50", self.start)
        self.start_btn.grid(row=0, column=0, padx=20)

        self.stop_btn = self.create_button("⏹ Stop", "#FF8C00", self.stop)
        self.stop_btn.grid(row=0, column=1, padx=20)

        self.exit_btn = self.create_button("❌ Exit", "#FF3333", self.exit_app)
        self.exit_btn.grid(row=0, column=2, padx=20)

    # Button creator with hover & click effect
    def create_button(self, text, color, command):
        btn = tk.Button(self.btn_frame, text=text, bg=color, fg="white",
                        font=("Arial", 14, "bold"), width=12, height=2,
                        relief="flat", activebackground="white", activeforeground=color,
                        command=command)
        # Hover effect
        btn.bind("<Enter>", lambda e: btn.config(bg="white", fg=color))
        btn.bind("<Leave>", lambda e: btn.config(bg=color, fg="white"))
        # Click animation
        btn.bind("<ButtonPress>", lambda e: btn.config(relief="sunken"))
        btn.bind("<ButtonRelease>", lambda e: btn.config(relief="flat"))
        return btn

    # Start camera
    def start(self):
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.process()

    # Stop camera
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.status.config(text="Status: Stopped")

    # Exit app
    def exit_app(self):
        self.stop()
        self.root.quit()
        self.root.destroy()

    # Main loop
    def process(self):
        if not self.running or not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        if latest_result and latest_result.hand_landmarks:
            for landmarks in latest_result.hand_landmarks:
                index = landmarks[8]
                thumb = landmarks[4]

                screen_w, screen_h = pyautogui.size()
                x = np.interp(index.x, (0, 1), (0, screen_w))
                y = np.interp(index.y, (0, 1), (0, screen_h))

                pyautogui.moveTo(x, y)
                self.status.config(text="Status: Moving")

                if get_dist(index, thumb) < 0.05:
                    if time.time() - self.last_click > 0.3:
                        pyautogui.click()
                        self.last_click = time.time()
                        self.status.config(text="Status: Click")

        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.process)


# --- RUN APP ---
root = tk.Tk()
# Replace 'background.png' with your image path
app = VirtualMouseApp(root, bg_image_path=r"C:\\Users\\User\\Downloads\\hand move.png")
try:
    root.mainloop()
except KeyboardInterrupt:
    print("App closed safely")

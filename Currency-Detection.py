import cv2
import torch
import pyttsx3
import time
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import threading
import queue
import json
import sounddevice as sd
import sys
from collections import Counter

from vosk import Model as VoskModel, KaldiRecognizer, SetLogLevel
SetLogLevel(-1)

sys.path.append(r'C:\Users\M.Adil\yolov5')
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

# TTS setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.7)

# Load model
model_path = 'C:\\best1.pt'
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found.")
    exit()

try:
    model = DetectMultiBackend(weights=model_path, device='cpu')
    model.model.float()
    model.eval()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit()

# Globals
cap = None
running = False
last_spoken = ""
last_speak_time = 0
speak_cooldown = 2
IMAGE_SIZE = 416
MIN_AGREEING_DETECTIONS = 2  # Count threshold

def speak_text(text):
    global last_spoken, last_speak_time
    current_time = time.time()
    if text != last_spoken or (current_time - last_speak_time) > speak_cooldown:
        engine.say(text)
        engine.runAndWait()
        last_spoken = text
        last_speak_time = current_time

def preprocess_frame(frame):
    img = letterbox(frame, IMAGE_SIZE, stride=32, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to('cpu').float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

def detect_and_display(frame):
    current_time = time.time()
    img = preprocess_frame(frame)
    pred = model(img)
    pred = non_max_suppression(pred, conf_thres=0.6, iou_thres=0.45, max_det=10)

    labels_detected = []

    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                label = model.names[int(cls)]
                labels_detected.append(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if labels_detected:
        counts = Counter(labels_detected)
        for label, count in counts.items():
            if count >= MIN_AGREEING_DETECTIONS:
                speak_text(f"Detected currency {label} rupees")
                break  # Speak only once per frame
    else:
        cv2.putText(frame, "No currency detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

def update_frame():
    global running, cap
    if running and cap is not None:
        ret, frame = cap.read()
        if ret:
            frame = detect_and_display(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
    root.after(10, update_frame)

def start_detection():
    global running, cap
    if not running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            return
        running = True
        speak_text("Real-time currency detection started")
        start_button.config(state="disabled")
        stop_button.config(state="normal")
        upload_button.config(state="disabled")

def stop_detection():
    global running, cap
    if running:
        running = False
        if cap is not None:
            cap.release()
            cap = None
        video_label.config(image='')
        speak_text("Real-time detection stopped")
        start_button.config(state="normal")
        stop_button.config(state="disabled")
        upload_button.config(state="normal")

def upload_photo():
    global running
    if running:
        messagebox.showwarning("Warning", "Please stop webcam detection before uploading a photo.")
        return

    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        frame = cv2.imread(file_path)
        if frame is None:
            messagebox.showerror("Error", "Could not load image.")
            return

        frame = detect_and_display(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

# Voice command: only for start and stop
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

def listen_for_voice_commands():
    model_path = r"C:\Users\M.Adil\Desktop\model"
    if not os.path.exists(model_path):
        print("❌ Please download the Vosk model and set the correct path.")
        return

    vosk_model = VoskModel(model_path)
    recognizer = KaldiRecognizer(vosk_model, 16000)

    with sd.RawInputStream(samplerate=16000, blocksize=4000, dtype='int16',
                           channels=1, callback=audio_callback):
        print("🎤 Listening offline (Vosk)...")
        while True:
            while not q.empty():
                data = q.get()
                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                    text = json.loads(result).get("text", "").lower()
                    print(f"🎙️ Recognized: {text}")
                    if "start" in text:
                        root.after(0, start_detection)
                    elif "stop" in text:
                        root.after(0, stop_detection)
            time.sleep(0.05)

# GUI setup
root = tk.Tk()
root.title("Currency Detection")
root.geometry("800x600")

video_label = tk.Label(root)
video_label.pack(pady=10)

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

start_button = tk.Button(button_frame, text="Start Detection", command=start_detection)
start_button.grid(row=0, column=0, padx=5)

stop_button = tk.Button(button_frame, text="Stop Detection", command=stop_detection, state="disabled")
stop_button.grid(row=0, column=1, padx=5)

upload_button = tk.Button(button_frame, text="Upload Currency Image", command=upload_photo)
upload_button.grid(row=0, column=2, padx=5)

root.after(10, update_frame)

# Start voice thread
voice_thread = threading.Thread(target=listen_for_voice_commands, daemon=True)
voice_thread.start()

def on_closing():
    stop_detection()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()

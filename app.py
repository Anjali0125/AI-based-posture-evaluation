import sys
import os
import time
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

# ── Auto-Installation of missing packages ──
def check_and_install_requirements():
    try:
        import numpy
        import pandas
        import cv2
        import mediapipe
        import sklearn
        import joblib
    except ImportError:
        print("Missing packages. Performing startup check and installing automatically...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("Auto-installation of packages complete!")
        except Exception as e:
            print(f"Warning: Auto-installation failed: {e}. Attempting to proceed anyway.")

check_and_install_requirements()

# ── Import libraries ──
import numpy as np
import pandas as pd
import cv2
import PIL.Image, PIL.ImageTk
import mediapipe as mp

from angle_calculator import extract_all_angles, landmarks_from_results
from pose_evaluator import (
    PoseEvaluator, 
    train_rf_model, 
    load_knowledgebase, 
    DATASET_DIR, 
    MODEL_DIR,
    ALL_FEATURES
)
from generate_dataset import generate_dataset_csv

# ── Auto-Training on Startup ──
def verify_and_train_model():
    csv_path = DATASET_DIR / "yoga_poses.csv"
    if not csv_path.exists():
        print("Seeding synthetic pose datasets...")
        generate_dataset_csv(filepath=csv_path)
    
    model_path = MODEL_DIR / "rf_model.pkl"
    if not model_path.exists():
        print("Training Random Forest classifier on startup...")
        train_rf_model(csv_path=str(csv_path))

verify_and_train_model()

# ── Custom Widgets ──
class CircularProgress(tk.Canvas):
    def __init__(self, parent, size=140, thickness=12, bg_color="#222130", **kwargs):
        super().__init__(parent, width=size, height=size, bg=bg_color, highlightthickness=0, **kwargs)
        self.size = size
        self.thickness = thickness
        self.bg_color = bg_color
        self.draw_empty()

    def draw_empty(self):
        self.delete("all")
        pad = self.thickness // 2 + 3
        # Background Track
        self.create_oval(pad, pad, self.size - pad, self.size - pad, outline="#333148", width=self.thickness)
        self.create_text(self.size // 2, self.size // 2, text="--%", fill="#9896B8", font=("Segoe UI", 18, "bold"))

    def set_value(self, score, color="#43D9AD"):
        self.delete("all")
        pad = self.thickness // 2 + 3
        # Background Track
        self.create_oval(pad, pad, self.size - pad, self.size - pad, outline="#333148", width=self.thickness)
        
        # Color arc
        extent = -(score / 100.0) * 359.9
        self.create_arc(pad, pad, self.size - pad, self.size - pad, start=90, extent=extent,
                        style="arc", outline=color, width=self.thickness)
        
        # Text label
        self.create_text(self.size // 2, self.size // 2, text=f"{int(score)}%", fill="#FFFFFF", font=("Segoe UI", 20, "bold"))


# ── Main Application Class ──
class YogaApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🧘 AI Yoga Posture Evaluation")
        self.geometry("1150x740")
        
        # Color Palette Configuration (Modern Dark Theme)
        self.configure(bg="#12121A")
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Global ttk styles configuration
        self.style.configure(".", background="#12121A", foreground="#FFFFFF", font=("Segoe UI", 10))
        self.style.configure("TFrame", background="#12121A")
        self.style.configure("Card.TFrame", background="#1C1B29", relief="flat", borderwidth=0)
        self.style.configure("Sidebar.TFrame", background="#0B0A11")
        
        # Label Styles
        self.style.configure("TLabel", background="#12121A", foreground="#FFFFFF")
        self.style.configure("Title.TLabel", background="#12121A", foreground="#7F5AF0", font=("Segoe UI", 18, "bold"))
        self.style.configure("CardTitle.TLabel", background="#1C1B29", foreground="#7F5AF0", font=("Segoe UI", 13, "bold"))
        self.style.configure("Subtitle.TLabel", background="#1C1B29", foreground="#94A1B2", font=("Segoe UI", 9))
        self.style.configure("SidebarLabel.TLabel", background="#0B0A11", foreground="#94A1B2", font=("Segoe UI", 9, "bold"))
        
        # Button Styles
        self.style.configure("TButton", background="#7F5AF0", foreground="#FFFFFF", borderwidth=0, padding=10, font=("Segoe UI", 10, "bold"))
        self.style.map("TButton", background=[("active", "#6246EA"), ("disabled", "#2B2A3D")])
        self.style.configure("Nav.TButton", background="#0B0A11", foreground="#94A1B2", borderwidth=0, padding=12, font=("Segoe UI", 11, "bold"))
        self.style.map("Nav.TButton", background=[("active", "#12121A"), ("selected", "#7F5AF0")], foreground=[("active", "#FFFFFF"), ("selected", "#FFFFFF")])
        self.style.configure("Stop.TButton", background="#FF5C5C", foreground="#FFFFFF", padding=10, font=("Segoe UI", 10, "bold"))
        self.style.map("Stop.TButton", background=[("active", "#E04F4F")])
        
        # Combobox / Dropdown configuration
        self.style.configure("TCombobox", fieldbackground="#1C1B29", background="#7F5AF0", foreground="#FFFFFF")
        
        # Load Pose Evaluator
        self.evaluator = PoseEvaluator()
        
        # Feed/Media states
        self.cap = None
        self.camera_active = False
        self.video_playing = False
        
        # Create Layout
        self._create_widgets()
        
        # Load preconfigured poses
        self.refresh_pose_list()
        
        # Bind Close Event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _create_widgets(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # ── SIDEBAR (Left Panel) ──
        self.sidebar = ttk.Frame(self, style="Sidebar.TFrame", width=240)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        
        brand_lbl = ttk.Label(self.sidebar, text="🧘 YogaAI", font=("Segoe UI", 20, "bold"), background="#0B0A11", foreground="#7F5AF0")
        brand_lbl.pack(pady=(35, 10), padx=20, anchor="w")
        
        divider = tk.Frame(self.sidebar, height=1, bg="#2B2A3D")
        divider.pack(fill="x", padx=20, pady=(0, 25))
        
        # Sidebar Menu
        self.nav_live = ttk.Button(self.sidebar, text="📸 Live Evaluation", style="Nav.TButton", command=lambda: self.switch_tab("live"))
        self.nav_live.pack(fill="x", padx=15, pady=6)
        
        self.nav_media = ttk.Button(self.sidebar, text="📁 Upload Media", style="Nav.TButton", command=lambda: self.switch_tab("media"))
        self.nav_media.pack(fill="x", padx=15, pady=6)
        
        # Target Pose Picker on Sidebar
        ttk.Label(self.sidebar, text="TARGET POSE", style="SidebarLabel.TLabel").pack(pady=(45, 5), padx=20, anchor="w")
        self.pose_selector = ttk.Combobox(self.sidebar, state="readonly")
        self.pose_selector.pack(fill="x", padx=20, pady=5)
        
        # Footer
        footer_frame = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        footer_frame.pack(side="bottom", fill="x", pady=25, padx=20)
        ttk.Label(footer_frame, text="Real-Time Analysis Mode", font=("Segoe UI", 8), foreground="#5C5B77", background="#0B0A11").pack(anchor="w")
        ttk.Label(footer_frame, text="Active model: Random Forest", font=("Segoe UI", 8), foreground="#5C5B77", background="#0B0A11").pack(anchor="w")
        
        # ── MAIN AREA (Right Panel) ──
        self.main_container = ttk.Frame(self)
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)
        
        self.tabs = {}
        
        self._init_live_tab()
        self._init_media_tab()
        
        # Default Tab
        self.switch_tab("live")

    def switch_tab(self, tab_name):
        self.stop_all_feeds()
        
        for tab in self.tabs.values():
            tab.grid_remove()
            
        self.tabs[tab_name].grid(row=0, column=0, sticky="nsew")
        
        self.nav_live.state(["!selected"])
        self.nav_media.state(["!selected"])
        
        if tab_name == "live":
            self.nav_live.state(["selected"])
        elif tab_name == "media":
            self.nav_media.state(["selected"])

    def stop_all_feeds(self):
        self.camera_active = False
        self.video_playing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.live_gauge.draw_empty()
        self.media_gauge.draw_empty()

    def refresh_pose_list(self):
        kb = load_knowledgebase()
        pose_keys = ["auto"] + list(kb.keys())
        self.pose_selector["values"] = pose_keys
        self.pose_selector.set("auto")

    # ── TAB: LIVE EVALUATION ──
    def _init_live_tab(self):
        tab = ttk.Frame(self.main_container)
        self.tabs["live"] = tab
        
        tab.grid_columnconfigure(0, weight=3)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        
        # Left Panel (Webcam frame Card)
        left_frame = ttk.Frame(tab)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_rowconfigure(0, weight=1)
        
        # Video Canvas Container Card
        canvas_card = ttk.Frame(left_frame, style="Card.TFrame")
        canvas_card.grid(row=0, column=0, sticky="nsew")
        canvas_card.grid_columnconfigure(0, weight=1)
        canvas_card.grid_rowconfigure(1, weight=1)
        
        # Card Header
        lbl_head = ttk.Label(canvas_card, text="Webcam Live Video HUD Overlay", style="CardTitle.TLabel")
        lbl_head.grid(row=0, column=0, sticky="w", padx=20, pady=(15, 10))
        
        # Canvas
        self.live_canvas = tk.Canvas(canvas_card, bg="#0F0F16", highlightthickness=0)
        self.live_canvas.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        
        # Controls Frame
        ctrl_frame = ttk.Frame(left_frame)
        ctrl_frame.grid(row=1, column=0, sticky="ew", pady=(15, 0))
        
        self.btn_toggle_cam = ttk.Button(ctrl_frame, text="▶ Start Live Webcam", command=self.toggle_webcam)
        self.btn_toggle_cam.pack(side="left")
        
        # Right Panel (Feedback HUD)
        right_frame = ttk.Frame(tab, style="Card.TFrame")
        right_frame.grid(row=0, column=1, sticky="nsew")
        
        ttk.Label(right_frame, text="Live Performance HUD", style="CardTitle.TLabel").pack(pady=(20, 5), padx=20, anchor="w")
        ttk.Label(right_frame, text="Real-time Pose Alignment Metrics", style="Subtitle.TLabel").pack(padx=20, anchor="w", pady=(0, 15))
        
        # Gauge Placement
        gauge_container = ttk.Frame(right_frame, style="Card.TFrame")
        gauge_container.pack(fill="x", padx=20, pady=10)
        
        self.live_gauge = CircularProgress(gauge_container, size=150, thickness=12, bg_color="#1C1B29")
        self.live_gauge.pack(pady=5)
        
        self.live_pose_lbl = ttk.Label(right_frame, text="Pose: --", font=("Segoe UI", 12, "bold"), background="#1C1B29", foreground="#7F5AF0")
        self.live_pose_lbl.pack(pady=10, padx=20, anchor="w")
        
        self.live_quality_lbl = ttk.Label(right_frame, text="Alignment: No body detected", font=("Segoe UI", 10, "bold"), background="#1C1B29", foreground="#94A1B2")
        self.live_quality_lbl.pack(pady=(0, 15), padx=20, anchor="w")
        
        # Feedback Text Panel
        ttk.Label(right_frame, text="CORRECTION ADVICE", font=("Segoe UI", 9, "bold"), background="#1C1B29", foreground="#94A1B2").pack(padx=20, anchor="w")
        self.live_fb_text = tk.Text(right_frame, bg="#12121A", fg="#FFFFFF", insertbackground="white", relief="flat", wrap="word", height=15, font=("Segoe UI", 10))
        self.live_fb_text.pack(fill="both", expand=True, padx=20, pady=(5, 20))
        self.live_fb_text.config(state="disabled")

    def toggle_webcam(self):
        if self.camera_active:
            self.stop_all_feeds()
            self.btn_toggle_cam.configure(text="▶ Start Live Webcam")
            self.live_canvas.delete("all")
        else:
            self.stop_all_feeds()
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam feed.")
                self.cap = None
                return
            self.camera_active = True
            self.btn_toggle_cam.configure(text="⏹ Stop Live Webcam")
            self.webcam_loop()

    def webcam_loop(self):
        if not self.camera_active or self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.after(10, self.webcam_loop)
            return

        frame = cv2.flip(frame, 1)
        tgt = self.pose_selector.get()
        result = self.evaluator.evaluate(frame, target_pose=tgt)
        
        if result is not None:
            disp_frame = result.skeleton_image
            
            # Update HUD Widgets
            self.live_gauge.set_value(result.score, self.get_score_color(result.score))
            self.live_pose_lbl.configure(text=f"{result.display_name} {result.emoji}")
            self.live_quality_lbl.configure(text=f"Alignment: {result.quality}", foreground=self.get_score_color(result.score))
            
            # Update Correction log
            self.live_fb_text.config(state="normal")
            self.live_fb_text.delete("1.0", tk.END)
            for fb in result.feedback:
                self.live_fb_text.insert(tk.END, fb + "\n")
            self.live_fb_text.config(state="disabled")
        else:
            disp_frame = frame
            self.live_gauge.draw_empty()
            self.live_pose_lbl.configure(text="Pose: --")
            self.live_quality_lbl.configure(text="Alignment: No body detected", foreground="#94A1B2")
            
            self.live_fb_text.config(state="normal")
            self.live_fb_text.delete("1.0", tk.END)
            self.live_fb_text.config(state="disabled")

        self.draw_frame_to_canvas(disp_frame, self.live_canvas)
        self.after(10, self.webcam_loop)

    # ── TAB: MEDIA UPLOAD ──
    def _init_media_tab(self):
        tab = ttk.Frame(self.main_container)
        self.tabs["media"] = tab
        
        tab.grid_columnconfigure(0, weight=3)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        
        left_frame = ttk.Frame(tab)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_rowconfigure(0, weight=1)
        
        canvas_card = ttk.Frame(left_frame, style="Card.TFrame")
        canvas_card.grid(row=0, column=0, sticky="nsew")
        canvas_card.grid_columnconfigure(0, weight=1)
        canvas_card.grid_rowconfigure(1, weight=1)
        
        lbl_head = ttk.Label(canvas_card, text="Media File Evaluation Viewer", style="CardTitle.TLabel")
        lbl_head.grid(row=0, column=0, sticky="w", padx=20, pady=(15, 10))
        
        self.media_canvas = tk.Canvas(canvas_card, bg="#0F0F16", highlightthickness=0)
        self.media_canvas.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        
        ctrl_frame = ttk.Frame(left_frame)
        ctrl_frame.grid(row=1, column=0, sticky="ew", pady=(15, 0))
        
        ttk.Button(ctrl_frame, text="🖼️ Upload Image", command=self.upload_image).pack(side="left", padx=(0, 10))
        ttk.Button(ctrl_frame, text="🎥 Upload Video", command=self.upload_video).pack(side="left", padx=10)
        self.btn_stop_video = ttk.Button(ctrl_frame, text="⏹ Stop Video Stream", style="Stop.TButton", command=self.stop_all_feeds)
        self.btn_stop_video.pack(side="left", padx=10)
        
        # Right Panel (HUD Stats)
        right_frame = ttk.Frame(tab, style="Card.TFrame")
        right_frame.grid(row=0, column=1, sticky="nsew")
        
        ttk.Label(right_frame, text="Evaluation Summary", style="CardTitle.TLabel").pack(pady=(20, 5), padx=20, anchor="w")
        ttk.Label(right_frame, text="Processed Posture Diagnostics", style="Subtitle.TLabel").pack(padx=20, anchor="w", pady=(0, 15))
        
        # Score Gauge
        gauge_container = ttk.Frame(right_frame, style="Card.TFrame")
        gauge_container.pack(fill="x", padx=20, pady=10)
        
        self.media_gauge = CircularProgress(gauge_container, size=150, thickness=12, bg_color="#1C1B29")
        self.media_gauge.pack(pady=5)
        
        self.media_pose_lbl = ttk.Label(right_frame, text="Pose: --", font=("Segoe UI", 12, "bold"), background="#1C1B29", foreground="#7F5AF0")
        self.media_pose_lbl.pack(pady=10, padx=20, anchor="w")
        
        self.media_quality_lbl = ttk.Label(right_frame, text="Alignment: No media loaded", font=("Segoe UI", 10, "bold"), background="#1C1B29", foreground="#94A1B2")
        self.media_quality_lbl.pack(pady=(0, 15), padx=20, anchor="w")
        
        # Feedback Text Panel
        ttk.Label(right_frame, text="ALIGNMENT HINTS", font=("Segoe UI", 9, "bold"), background="#1C1B29", foreground="#94A1B2").pack(padx=20, anchor="w")
        self.media_fb_text = tk.Text(right_frame, bg="#12121A", fg="#FFFFFF", insertbackground="white", relief="flat", wrap="word", height=15, font=("Segoe UI", 10))
        self.media_fb_text.pack(fill="both", expand=True, padx=20, pady=(5, 20))
        self.media_fb_text.config(state="disabled")

    def upload_image(self):
        self.stop_all_feeds()
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        
        frame = cv2.imread(file_path)
        if frame is None:
            messagebox.showerror("Error", "Could not load image file.")
            return
            
        tgt = self.pose_selector.get()
        result = self.evaluator.evaluate(frame, target_pose=tgt)
        
        if result is not None:
            self.draw_frame_to_canvas(result.skeleton_image, self.media_canvas)
            self.media_gauge.set_value(result.score, self.get_score_color(result.score))
            self.media_pose_lbl.configure(text=f"{result.display_name} {result.emoji}")
            self.media_quality_lbl.configure(text=f"Alignment: {result.quality}", foreground=self.get_score_color(result.score))
            
            self.media_fb_text.config(state="normal")
            self.media_fb_text.delete("1.0", tk.END)
            for fb in result.feedback:
                self.media_fb_text.insert(tk.END, fb + "\n")
            self.media_fb_text.config(state="disabled")
        else:
            self.draw_frame_to_canvas(frame, self.media_canvas)
            self.media_gauge.draw_empty()
            self.media_pose_lbl.configure(text="Pose: --")
            self.media_quality_lbl.configure(text="Alignment: No body detected", foreground="#94A1B2")
            
            self.media_fb_text.config(state="normal")
            self.media_fb_text.delete("1.0", tk.END)
            self.media_fb_text.config(state="disabled")
            messagebox.showwarning("Warning", "No pose landmarks detected in image.")

    def upload_video(self):
        self.stop_all_feeds()
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if not file_path:
            return
            
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open video file.")
            self.cap = None
            return
            
        self.video_playing = True
        self.video_loop()

    def video_loop(self):
        if not self.video_playing or self.cap is None:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                self.stop_all_feeds()
                return

        tgt = self.pose_selector.get()
        result = self.evaluator.evaluate(frame, target_pose=tgt)
        
        if result is not None:
            disp_frame = result.skeleton_image
            self.media_gauge.set_value(result.score, self.get_score_color(result.score))
            self.media_pose_lbl.configure(text=f"{result.display_name} {result.emoji}")
            self.media_quality_lbl.configure(text=f"Alignment: {result.quality}", foreground=self.get_score_color(result.score))
            
            self.media_fb_text.config(state="normal")
            self.media_fb_text.delete("1.0", tk.END)
            for fb in result.feedback:
                self.media_fb_text.insert(tk.END, fb + "\n")
            self.media_fb_text.config(state="disabled")
        else:
            disp_frame = frame
            self.media_gauge.draw_empty()
            self.media_pose_lbl.configure(text="Pose: --")
            self.media_quality_lbl.configure(text="Alignment: No body detected", foreground="#94A1B2")

        self.draw_frame_to_canvas(disp_frame, self.media_canvas)
        self.after(30, self.video_loop)

    # ── Utility Helpers ──
    def draw_frame_to_canvas(self, frame, canvas):
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(rgb_img)
        
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw > 10 and ch > 10:
            pil_img = pil_img.resize((cw, ch), PIL.Image.Resampling.LANCZOS)
            
        self.photo = PIL.ImageTk.PhotoImage(image=pil_img)
        canvas.delete("all")
        canvas.create_image(0, 0, image=self.photo, anchor="nw")

    def get_score_color(self, score):
        if score >= 88: return "#2CB67D"   # Mint green
        if score >= 72: return "#7F5AF0"   # Purple
        if score >= 50: return "#FFB547"   # Amber orange
        return "#FF5C5C"                   # Coral red

    def on_closing(self):
        self.stop_all_feeds()
        self.destroy()

if __name__ == "__main__":
    app = YogaApp()
    app.mainloop()

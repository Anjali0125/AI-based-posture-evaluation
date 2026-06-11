#  AI-Based Yoga Posture Evaluation System

A native cross-platform desktop application designed to track, evaluate, and provide real-time correction feedback for yoga poses using Google MediaPipe Pose and Scikit-Learn Random Forest.

## Key Features

- **Real-Time Landmark Tracking**: Localizes 33 body coordinates from a live webcam feed or uploaded media (images/videos) using Google MediaPipe Pose.
- **8-Angle Feature Classification**: Classifies postures (Tree, Triangle, Mountain, and Chair poses) using a Scikit-Learn Random Forest model trained on 8 key joint angles (`left_elbow`, `right_elbow`, `left_shoulder`, `right_shoulder`, `left_hip`, `right_hip`, `left_knee`, and `right_knee`).
- **Dynamic Alignment Scoring**: Computes an accuracy score out of 100 based on the joint angle deviations from ideal target templates
- **Real-Time Corrections HUD**: Overlays visual cues on the video canvas. Highlights joints as **green** (correct alignment) or **red** (needs adjustment) with live joint angle readouts and tailored correction tips.
- **Native Cross-Platform UI**: Built with a sleek dark-slate theme using standard Python `tkinter/ttk` widgets, featuring custom-drawn circular progress score rings.
- **Auto-Installation**: Checks for required packages on startup and installs them automatically from `requirements.txt` if they are missing.
- **Auto-Training on Startup**: Checks for the synthetic pose dataset and trained models on boot. If missing, it automatically generates a 1,200-sample Gaussian-distributed dataset in the `data/` folder and trains the Random Forest model instantly.

---

## Folder Structure

```
yoga evaluation/
├── data/
│   └── yoga_poses.csv          # Seeded synthetic and recorded CSV dataset
├── models/
│   ├── rf_model.pkl            # Trained Random Forest classifier
│   ├── scaler.pkl              # Fitted StandardScaler
│   ├── label_encoder.pkl       # Fitted LabelEncoder
│   └── metadata.json           # Model training information
├── angle_calculator.py         # 2D Joint angle calculation routines
├── pose_evaluator.py           # Evaluation pipeline, scoring, and HUD overlays
├── generate_dataset.py         # Seeds Gaussian-based synthetic angles data
├── app.py                      # Main entrypoint containing the Tkinter GUI app
└── requirements.txt            # Project dependencies list
```

---

## Requirements

The project requires Python 3.8+ and uses the following main dependencies:
- `numpy`
- `pandas`
- `opencv-python`
- `mediapipe` (<=0.10.14)
- `scikit-learn`
- `joblib`
- `tkinter` (Standard Python Library)

---

## Getting Started

### 1. Run the Application
Start the desktop application using your terminal:
```bash
python app.py
```

*Note: On first boot, the application will check for missing libraries, download them, seed the posture dataset inside `data/yoga_poses.csv`, and train the classifier automatically.*

### 2. Live Evaluation Mode
- Click the **📸 Live Evaluation** tab.
- Click **▶ Start Live Webcam** to launch your webcam stream.
- Select a target pose from the **Target Pose** dropdown (or leave it on `auto` for the Random Forest classifier to identify it automatically).
- Align your full body in the frame. Incorrect joint angles will highlight in **red** on the HUD feed, and the correction panel will display advice.

### 3. Media Upload Mode
- Click the Upload Media tab.
- Click Upload Image or Upload Video to load files.
- The system will process the file, draw the skeleton overlay, evaluate the posture, and display diagnostic scores.

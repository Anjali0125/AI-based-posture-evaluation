from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

from angle_calculator import extract_all_angles, landmarks_from_results

log = logging.getLogger(__name__)

# Directory paths
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
DATASET_DIR = Path("data")
DATASET_DIR.mkdir(exist_ok=True)

DEFAULT_POSES = {
    "Vrikshasana": {
        "display_name": "Vrikshasana (Tree Pose)",
        "emoji": "🌳",
        "description": "Standing on one leg, other foot on inner thigh, hands in prayer or overhead.",
        "angles": {
            "right_knee": 180.0,
            "right_hip": 175.0,
            "left_knee": 50.0,
            "left_hip": 100.0,
            "left_shoulder": 160.0,
            "right_shoulder": 160.0,
            "left_elbow": 175.0,
            "right_elbow": 175.0,
        },
        "keywords": {
            "right_knee": "Standing knee (should be straight)",
            "right_hip": "Standing hip (extended)",
            "left_knee": "Bent knee (deep fold)",
            "left_hip": "Raised hip angle",
            "left_shoulder": "Left arm overhead",
            "right_shoulder": "Right arm overhead",
            "left_elbow": "Left arm straight",
            "right_elbow": "Right arm straight",
        }
    },
    "Trikonasana": {
        "display_name": "Trikonasana (Triangle Pose)",
        "emoji": "📐",
        "description": "Wide-stance side bend, both legs straight, arms open vertically.",
        "angles": {
            "left_knee": 175.0,
            "right_knee": 175.0,
            "left_hip": 65.0,
            "right_hip": 110.0,
            "left_shoulder": 170.0,
            "right_shoulder": 170.0,
            "left_elbow": 175.0,
            "right_elbow": 175.0,
        },
        "keywords": {
            "left_knee": "Left knee straight",
            "right_knee": "Right knee straight",
            "left_hip": "Hip side bend",
            "right_hip": "Hip side extension",
            "left_shoulder": "Left arm vertical",
            "right_shoulder": "Right arm vertical",
        }
    },
    "Tadasana": {
        "display_name": "Tadasana (Mountain Pose)",
        "emoji": "🏔️",
        "description": "Upright standing pose, feet together, arms at sides, joints extended.",
        "angles": {
            "left_knee": 175.0,
            "right_knee": 175.0,
            "left_hip": 175.0,
            "right_hip": 175.0,
            "left_shoulder": 15.0,
            "right_shoulder": 15.0,
            "left_elbow": 175.0,
            "right_elbow": 175.0,
        },
        "keywords": {
            "left_knee": "Left knee straight",
            "right_knee": "Right knee straight",
            "left_hip": "Left hip extended",
            "right_hip": "Right hip extended",
            "left_shoulder": "Left arm at side",
            "right_shoulder": "Right arm at side",
        }
    },
    "Utkatasana": {
        "display_name": "Utkatasana (Chair Pose)",
        "emoji": "🪑",
        "description": "Squat position, knees and hips bent at ~110°, arms extended up.",
        "angles": {
            "left_knee": 110.0,
            "right_knee": 110.0,
            "left_hip": 110.0,
            "right_hip": 110.0,
            "left_shoulder": 155.0,
            "right_shoulder": 155.0,
            "left_elbow": 172.0,
            "right_elbow": 172.0,
        },
        "keywords": {
            "left_knee": "Left knee bent",
            "right_knee": "Right knee bent",
            "left_hip": "Left hip bent",
            "right_hip": "Right hip bent",
            "left_shoulder": "Left arm raised",
            "right_shoulder": "Right arm raised",
        }
    }
}

KB_FILE = Path("knowledgebase.json")

def load_knowledgebase() -> dict:
    if KB_FILE.exists():
        try:
            with open(KB_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            log.warning("Could not read knowledgebase.json: %s", e)
    # Return default and save it
    save_knowledgebase(DEFAULT_POSES)
    return DEFAULT_POSES

def save_knowledgebase(data: dict):
    try:
        with open(KB_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        log.warning("Could not save knowledgebase.json: %s", e)

REFERENCE_POSES = load_knowledgebase()

EXCELLENT_THRESHOLD = 12
GOOD_THRESHOLD = 25
FAIR_THRESHOLD = 40

ALL_FEATURES = [
    "left_elbow",
    "right_elbow",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
]

@dataclass
class EvaluationResult:
    pose_name: str
    display_name: str
    emoji: str
    score: float
    quality: str
    angles: dict[str, float]
    feedback: list[str]
    joint_table: pd.DataFrame
    detected_image: Optional[np.ndarray] = None
    skeleton_image: Optional[np.ndarray] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

def train_rf_model(csv_path: str = "data/yoga_poses.csv") -> bool:
    """Train the RandomForestClassifier using the CSV dataset and save model assets."""
    try:
        if not os.path.exists(csv_path):
            log.warning(f"CSV path {csv_path} does not exist. Cannot train.")
            return False

        df = pd.read_csv(csv_path)
        if df.empty or "pose" not in df.columns:
            return False

        # Drop NaN values if any
        df = df.dropna()

        X = df[ALL_FEATURES]
        y = df["pose"]

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_scaled, y_encoded)

        joblib.dump(clf, MODEL_DIR / "rf_model.pkl")
        joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
        joblib.dump(le, MODEL_DIR / "label_encoder.pkl")

        metadata = {
            "features": ALL_FEATURES,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples": len(df)
        }
        with open(MODEL_DIR / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        return True
    except Exception as e:
        log.error("Error training model: %s", e)
        return False

class PoseEvaluator:
    def __init__(self, detection_confidence: float = 0.5, model_complexity: int = 1):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self._pose = self.mp_pose.Pose(
            static_image_mode=False, # Optimized for streaming/videos
            min_detection_confidence=detection_confidence,
            model_complexity=model_complexity,
        )
        self.clf = None
        self.scaler = None
        self.le = None
        self.load_model()

    def load_model(self):
        try:
            model_path = MODEL_DIR / "rf_model.pkl"
            scaler_path = MODEL_DIR / "scaler.pkl"
            le_path = MODEL_DIR / "label_encoder.pkl"
            if model_path.exists() and scaler_path.exists() and le_path.exists():
                self.clf = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.le = joblib.load(le_path)
        except Exception as e:
            log.warning("Could not load ML model: %s", e)

    def evaluate(self, image_bgr: np.ndarray, target_pose: str = "auto") -> Optional[EvaluationResult]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(image_rgb)

        if not results.pose_landmarks:
            return None

        h, w, _ = image_bgr.shape
        landmarks = landmarks_from_results(results, w, h)
        angles = extract_all_angles(landmarks)

        skeleton_img = self._draw_skeleton(image_bgr.copy(), results)

        global REFERENCE_POSES
        REFERENCE_POSES = load_knowledgebase() # Reload to reflect any modifications

        # Posture Classification
        if target_pose == "auto":
            pose_key, confidence = self._classify_pose(angles)
        else:
            pose_key = target_pose
            confidence = None

        if pose_key not in REFERENCE_POSES:
            pose_key = list(REFERENCE_POSES.keys())[0]

        ref = REFERENCE_POSES[pose_key]
        score, joint_df, feedback = self._compute_score_and_feedback(angles, ref)
        quality = self._quality_label(score)

        # Draw real-time alignment corrections on the skeleton image
        self._overlay_corrections_hud(skeleton_img, angles, ref, landmarks)

        return EvaluationResult(
            pose_name=pose_key,
            display_name=ref.get("display_name", pose_key),
            emoji=ref.get("emoji", "🧘"),
            score=score,
            quality=quality,
            angles=angles,
            feedback=feedback,
            joint_table=joint_df,
            detected_image=image_bgr,
            skeleton_image=skeleton_img,
        )

    def _draw_skeleton(self, image: np.ndarray, results) -> np.ndarray:
        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.pose_landmarks,
            connections=self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(0, 255, 170), thickness=3, circle_radius=5
            ),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(255, 100, 0), thickness=2
            ),
        )
        return image

    def _classify_pose(self, angles: dict) -> Tuple[str, float]:
        if self.clf is not None and self.scaler is not None and self.le is not None:
            try:
                feat_df = pd.DataFrame([[angles.get(f, 90.0) for f in ALL_FEATURES]], columns=ALL_FEATURES)
                feat_scaled = self.scaler.transform(feat_df)
                pred_idx = self.clf.predict(feat_scaled)[0]
                proba = self.clf.predict_proba(feat_scaled)[0]
                pose_name = self.le.inverse_transform([pred_idx])[0]
                confidence = float(proba[pred_idx])
                return pose_name, confidence
            except Exception as e:
                log.warning("ML classification failed: %s", e)

        # Fallback to rule-based MAE matching
        scores = {}
        for pose_key, ref in REFERENCE_POSES.items():
            errors = [
                abs(angles[a] - ideal)
                for a, ideal in ref["angles"].items()
                if a in angles
            ]
            if errors:
                scores[pose_key] = sum(errors) / len(errors)

        if not scores:
            return list(REFERENCE_POSES.keys())[0], 0.0

        best_pose = min(scores, key=scores.get)
        confidence = max(0.0, 1.0 - scores[best_pose] / 90.0)
        return best_pose, confidence

    def _compute_score_and_feedback(self, angles: dict, ref: dict) -> Tuple[float, pd.DataFrame, list[str]]:
        ref_angles = ref["angles"]
        keywords = ref.get("keywords", {})
        rows = []
        errors = []
        feedback = []

        for angle_name, ideal in ref_angles.items():
            user_val = angles.get(angle_name)
            if user_val is None:
                continue

            diff = abs(user_val - ideal)
            errors.append(diff)
            label = keywords.get(angle_name, angle_name.replace("_", " ").title())

            if diff <= EXCELLENT_THRESHOLD:
                status = "✅ Excellent"
                feedback.append(f"✓ {label}: excellent alignment")
            elif diff <= GOOD_THRESHOLD:
                status = "✔️ Good"
                feedback.append(f"✓ {label}: good - small adjustment needed")
            elif diff <= FAIR_THRESHOLD:
                status = "⚠️ Fair"
                hint = _direction_hint(angle_name, user_val, ideal)
                feedback.append(f"✗ {label}: {hint}")
            else:
                status = "❌ Needs Work"
                hint = _direction_hint(angle_name, user_val, ideal)
                feedback.append(f"✗ {label}: large deviation - {hint}")

            rows.append({
                "Joint / Angle": label,
                "Your Angle (°)": round(user_val, 1),
                "Ideal (°)": ideal,
                "Difference (°)": round(diff, 1),
                "Status": status,
            })

        if errors:
            mean_err = sum(errors) / len(errors)
            score = max(0.0, min(100.0, 100.0 - mean_err)) # Score = 100 - Average Angle Error
        else:
            score = 0.0

        if score >= 88:
            feedback.insert(0, "✓ Outstanding posture! Keep holding it.")
        elif score >= 72:
            feedback.insert(0, "✓ Good effort - make small adjustments.")
        elif score >= 50:
            feedback.insert(0, "⚠️ Keep practicing - hold the pose and stabilize.")
        else:
            feedback.insert(0, "✗ Significant correction needed - adjust red joints.")

        df = pd.DataFrame(rows)
        return round(score, 1), df, feedback

    def _overlay_corrections_hud(self, image: np.ndarray, angles: dict, ref: dict, landmarks: list):
        """Displays visual overlays highlighting joints as green or red with metrics on the camera frame."""
        PL = self.mp_pose.PoseLandmark
        
        joint_landmarks = {
            "left_elbow": PL.LEFT_ELBOW.value,
            "right_elbow": PL.RIGHT_ELBOW.value,
            "left_shoulder": PL.LEFT_SHOULDER.value,
            "right_shoulder": PL.RIGHT_SHOULDER.value,
            "left_knee": PL.LEFT_KNEE.value,
            "right_knee": PL.RIGHT_KNEE.value,
            "left_hip": PL.LEFT_HIP.value,
            "right_hip": PL.RIGHT_HIP.value,
        }

        for angle_name, ideal in ref["angles"].items():
            user_val = angles.get(angle_name)
            if user_val is None:
                continue

            diff = abs(user_val - ideal)
            color = (0, 255, 0) if diff <= GOOD_THRESHOLD else (0, 0, 255) # Green vs Red

            # Find matching landmark index to draw circles/text on
            lm_idx = joint_landmarks.get(angle_name)
            if lm_idx is not None and lm_idx < len(landmarks):
                x, y, _ = landmarks[lm_idx]
                
                # Draw outer glow circle
                cv2.circle(image, (x, y), 12, color, 2)
                cv2.circle(image, (x, y), 4, color, -1)
                
                # Display current and target angle
                text = f"{user_val:.0f}/{ideal:.0f}"
                cv2.putText(
                    image, text, (x + 15, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
                )

    def _quality_label(self, score: float) -> str:
        if score >= 88:
            return "Excellent"
        elif score >= 72:
            return "Good"
        elif score >= 50:
            return "Fair"
        else:
            return "Needs Work"

def _direction_hint(angle_name: str, user_val: float, ideal: float) -> str:
    diff = user_val - ideal
    if "knee" in angle_name:
        return "bend knee more" if diff < 0 else "straighten knee more"
    elif "elbow" in angle_name:
        return "straighten arm more" if diff < 0 else "bend elbow slightly"
    elif "shoulder" in angle_name:
        return "raise arm higher" if diff < 0 else "lower arm slightly"
    elif "hip" in angle_name:
        return "open hip more" if diff < 0 else "bring hip inward"
    elif "neck" in angle_name:
        return "lift head forward" if diff < 0 else "tuck chin gently"
    elif "wrist" in angle_name:
        return "rotate wrist inward" if diff < 0 else "rotate wrist outward"
    elif "ankle" in angle_name or "inter" in angle_name:
        return "widen your stance" if diff < 0 else "narrow your stance"
    else:
        return f"adjust by {abs(diff):.0f}°"

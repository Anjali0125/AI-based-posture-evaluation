import os
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(seed=42)

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

POSE_DISTRIBUTIONS = {
    "Vrikshasana": { # Tree Pose
        "angles": {
            "right_knee":    (178.0, 5),   # Standing leg straight
            "right_hip":     (172.0, 6),   # Hip extended
            "left_knee":     (52.0,  8),   # Bent leg
            "left_hip":      (102.0, 10),  # Open hip
            "left_shoulder": (158.0, 8),   # Arms overhead
            "right_shoulder":(158.0, 8),
            "left_elbow":    (173.0, 6),
            "right_elbow":   (173.0, 6),
        }
    },
    "Trikonasana": { # Triangle Pose
        "angles": {
            "left_knee":     (174.0, 5),   # Both knees straight
            "right_knee":    (174.0, 5),
            "left_hip":      (68.0,  8),   # Side bend
            "right_hip":     (108.0, 10),
            "left_shoulder": (168.0, 7),   # Arms in line
            "right_shoulder":(168.0, 7),
            "left_elbow":    (174.0, 5),
            "right_elbow":   (174.0, 5),
        }
    },
    "Tadasana": { # Mountain Pose
        "angles": {
            "left_knee":     (175.0, 4),   # Straight standing
            "right_knee":    (175.0, 4),
            "left_hip":      (174.0, 5),
            "right_hip":     (174.0, 5),
            "left_shoulder": (15.0,  6),   # Arms at side
            "right_shoulder":(15.0,  6),
            "left_elbow":    (174.0, 5),
            "right_elbow":   (174.0, 5),
        }
    },
    "Utkatasana": { # Chair Pose
        "angles": {
            "left_knee":     (110.0, 8),   # Bent knees
            "right_knee":    (110.0, 8),
            "left_hip":      (110.0, 10),  # Hips bent
            "right_hip":     (110.0, 10),
            "left_shoulder": (155.0, 8),   # Arms up/forward
            "right_shoulder":(155.0, 8),
            "left_elbow":    (172.0, 6),
            "right_elbow":   (172.0, 6),
        }
    }
}

def _clip(val: float) -> float:
    return float(np.clip(val, 0.0, 180.0))

def generate_samples_for_pose(pose_name: str, config: dict, n_samples: int = 300) -> pd.DataFrame:
    angle_defs = config["angles"]
    rows = []
    for _ in range(n_samples):
        row = {"pose": pose_name}
        for feat in ALL_FEATURES:
            if feat in angle_defs:
                mean, sigma = angle_defs[feat]
            else:
                mean, sigma = 90.0, 25.0
            row[feat] = _clip(RNG.normal(mean, sigma))
        rows.append(row)
    return pd.DataFrame(rows)

def generate_dataset_csv(filepath=None, n_samples=300):
    if filepath is None:
        filepath = OUTPUT_DIR / "yoga_poses.csv"
    else:
        filepath = Path(filepath)
    
    all_dfs = []
    for pose_name, config in POSE_DISTRIBUTIONS.items():
        df = generate_samples_for_pose(pose_name, config, n_samples=n_samples)
        all_dfs.append(df)
        
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(filepath, index=False)
    print(f"Dataset generated at {filepath}")
    return filepath

if __name__ == "__main__":
    generate_dataset_csv()

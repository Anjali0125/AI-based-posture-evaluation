import math
import mediapipe as mp

mp_pose = mp.solutions.pose

def calculate_angle(landmark1: tuple, landmark2: tuple, landmark3: tuple) -> float:
    """
    Calculate the interior angle (in degrees) at landmark2 formed by landmark1 and landmark3.
    Result is always in the range [0°, 180°].
    """
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    radians = math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
    angle = math.degrees(radians)
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle

    return round(angle, 2)

def extract_all_angles(landmarks: list) -> dict:
    """
    Extract relevant joint angles from a list of 33 MediaPipe landmarks.
    """
    PL = mp_pose.PoseLandmark

    def lm(idx) -> tuple:
        return landmarks[idx.value]

    angles = {}

    # ── Elbow angles ────────────────────────────────────────────────────────
    # Vertex = elbow; rays go to shoulder and wrist
    angles["left_elbow"] = calculate_angle(
        lm(PL.LEFT_SHOULDER), lm(PL.LEFT_ELBOW), lm(PL.LEFT_WRIST)
    )
    angles["right_elbow"] = calculate_angle(
        lm(PL.RIGHT_SHOULDER), lm(PL.RIGHT_ELBOW), lm(PL.RIGHT_WRIST)
    )

    # ── Shoulder angles ─────────────────────────────────────────────────────
    # Vertex = shoulder; rays go to elbow and hip
    angles["left_shoulder"] = calculate_angle(
        lm(PL.LEFT_ELBOW), lm(PL.LEFT_SHOULDER), lm(PL.LEFT_HIP)
    )
    angles["right_shoulder"] = calculate_angle(
        lm(PL.RIGHT_ELBOW), lm(PL.RIGHT_SHOULDER), lm(PL.RIGHT_HIP)
    )

    # ── Knee angles ──────────────────────────────────────────────────────────
    # Vertex = knee; rays go to hip and ankle
    angles["left_knee"] = calculate_angle(
        lm(PL.LEFT_HIP), lm(PL.LEFT_KNEE), lm(PL.LEFT_ANKLE)
    )
    angles["right_knee"] = calculate_angle(
        lm(PL.RIGHT_HIP), lm(PL.RIGHT_KNEE), lm(PL.RIGHT_ANKLE)
    )

    # ── Hip angles ───────────────────────────────────────────────────────────
    # Vertex = hip; rays go to shoulder and knee
    angles["left_hip"] = calculate_angle(
        lm(PL.LEFT_SHOULDER), lm(PL.LEFT_HIP), lm(PL.LEFT_KNEE)
    )
    angles["right_hip"] = calculate_angle(
        lm(PL.RIGHT_SHOULDER), lm(PL.RIGHT_HIP), lm(PL.RIGHT_KNEE)
    )

    return angles

def landmarks_from_results(results, image_width: int, image_height: int) -> list:
    """Convert MediaPipe pose landmark results into (x, y, z) pixel values."""
    landmarks = []
    for lm in results.pose_landmarks.landmark:
        landmarks.append((
            int(lm.x * image_width),
            int(lm.y * image_height),
            lm.z * image_width,
        ))
    return landmarks

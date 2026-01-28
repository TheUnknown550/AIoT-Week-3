import cv2
import mediapipe as mp
import numpy as np
import time
import os
import math

# --- NEW API IMPORTS ---
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION & PATH FIX ---
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE_NAME = 'pose_landmarker_lite.task'
MODEL_PATH = os.path.join(script_dir, MODEL_FILE_NAME)

if not os.path.exists(MODEL_PATH):
    print(f"\nCRITICAL ERROR: Model file not found at {MODEL_PATH}")
    exit()

# --- DISTANCE THRESHOLDS (ADJUST THESE!) ---
# Note: MediaPipe coordinates are normalized (0.0 to 1.0).
# You might need to change these numbers depending on how far you stand.

# 1. Feet Distance (Ankle to Ankle)
FEET_DIST_OPEN = 0.4    # If dist > 0.4, legs are OPEN
FEET_DIST_CLOSE = 0.2   # If dist < 0.2, legs are CLOSED

# 2. Hand Height (Distance from Wrist to Hip)
# When doing jumping jacks, your hands go far away from your hips.
HAND_HIP_DIST_UP = 0.5   # If wrist is > 0.5 away from hip, arms are UP
HAND_HIP_DIST_DOWN = 0.2 # If wrist is < 0.2 away from hip, arms are DOWN

def calculate_distance(a, b):
    """
    Calculates the Euclidean distance between two landmarks.
    Returns a value roughly between 0.0 and 1.0.
    """
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    
    # Euclidean Distance Formula: sqrt((x2-x1)^2 + (y2-y1)^2)
    dist = np.linalg.norm(a - b)
    return dist

# --- SETUP MEDIAPIPE ---
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO
)

landmarker = PoseLandmarker.create_from_options(options)

# --- APP VARIABLES ---
counter = 0
stage = "down" 
feedback = "Stand in frame"
start_time = time.time()

cap = cv2.VideoCapture(1) # Change to 0 if 1 doesn't work

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 1. Prepare Image
    image = cv2.flip(frame, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    frame_timestamp_ms = int((time.time() - start_time) * 1000)

    # 2. Detect
    detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # 3. Logic
    if detection_result.pose_landmarks:
        lm = detection_result.pose_landmarks[0]
        
        # --- GET KEYPOINTS ---
        l_wrist = lm[15]
        r_wrist = lm[16]
        l_hip = lm[23]
        r_hip = lm[24]
        l_ankle = lm[27]
        r_ankle = lm[28]

        # --- CALCULATE DISTANCES ---
        
        # 1. Feet Spread (Distance between ankles)
        feet_spread = calculate_distance(l_ankle, r_ankle)
        
        # 2. Arm Lift (Distance from Wrist to Hip)
        # We average the left and right side
        l_arm_dist = calculate_distance(l_wrist, l_hip)
        r_arm_dist = calculate_distance(r_wrist, r_hip)
        avg_arm_dist = (l_arm_dist + r_arm_dist) / 2

        # --- COUNTING LOGIC ---
        
        # UP STATE: Feet wide apart AND Hands far from hips
        if feet_spread > FEET_DIST_OPEN and avg_arm_dist > HAND_HIP_DIST_UP:
            if stage == "down":
                stage = "up"
                counter += 1
                feedback = "GOOD REP"
        
        # DOWN STATE: Feet close together AND Hands close to hips
        elif feet_spread < FEET_DIST_CLOSE and avg_arm_dist < HAND_HIP_DIST_DOWN:
            stage = "down"
            feedback = "Ready"

        # --- DRAWING & DEBUG UI ---
        h, w, _ = image.shape
        
        # Draw Lines for visual feedback
        # Line between feet
        cx1, cy1 = int(l_ankle.x * w), int(l_ankle.y * h)
        cx2, cy2 = int(r_ankle.x * w), int(r_ankle.y * h)
        cv2.line(image, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
        
        # Draw skeleton dots
        for landmark in lm:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 5, (245, 117, 66), -1)

        # --- DISPLAY METRICS (Very helpful for tuning) ---
        # We display the current distance values so you can see what they are
        cv2.rectangle(image, (0,0), (400,160), (245,117,16), -1)
        
        cv2.putText(image, f'COUNT: {counter}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f'STAGE: {stage}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show real-time distance values
        cv2.putText(image, f'Feet Dist: {feet_spread:.2f}', (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(image, f'Arm Dist:  {avg_arm_dist:.2f}', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    cv2.imshow('Distance Jumper', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
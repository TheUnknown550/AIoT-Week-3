import cv2
import mediapipe as mp
import numpy as np
import time
import os

# --- NEW API IMPORTS ---
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION & PATH FIX ---
# This line forces Python to look for the model in the EXACT folder where this script is saved
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE_NAME = 'pose_landmarker_lite.task'
MODEL_PATH = os.path.join(script_dir, MODEL_FILE_NAME)

# Verify model exists
if not os.path.exists(MODEL_PATH):
    print(f"\nCRITICAL ERROR: Model file not found!")
    print(f"Looking for: {MODEL_PATH}")
    print(f"Please download '{MODEL_FILE_NAME}' and put it in that folder.\n")
    exit()

# --- ANGLES & LOGIC SETTINGS ---
# 1. Safety Check: How straight must limbs be? (Degrees)
# Lower = more lenient (easier to cheat), Higher = stricter form
ARM_STRAIGHT_THRESH = 130 
LEG_STRAIGHT_THRESH = 140

# 2. Position Check: What counts as "UP" vs "DOWN"?
ARM_UP_ANGLE = 130       # Arms must be higher than this
ARM_DOWN_ANGLE = 40      # Arms must be lower than this
LEG_SPREAD_UP = 172      # Legs must be WIDER than this (Angle < 172)
LEG_SPREAD_DOWN = 175    # Legs must be CLOSER than this (Angle > 175)

def calculate_angle(a, b, c):
    """Calculates angle between three points (a-b-c)."""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle

# --- SETUP MEDIAPIPE LANDMARKER ---
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

# --- MAIN LOOP ---
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Prepare Image
    image = cv2.flip(frame, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    # 2. Timestamp (Required for VIDEO mode)
    frame_timestamp_ms = int((time.time() - start_time) * 1000)

    # 3. Detect
    detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    
    # Revert to BGR for drawing
    image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # 4. Logic & Drawing
    if detection_result.pose_landmarks:
        # Get the first person detected
        lm = detection_result.pose_landmarks[0]
        
        # --- GET KEYPOINTS ---
        # Arms
        l_sh, l_el, l_wr = lm[11], lm[13], lm[15]
        r_sh, r_el, r_wr = lm[12], lm[14], lm[16]
        # Legs
        l_hp, l_kn, l_an = lm[23], lm[25], lm[27]
        r_hp, r_kn, r_an = lm[24], lm[26], lm[28]

        # --- SAFETY CHECK (STRAIGHTNESS) ---
        l_arm_straight = calculate_angle(l_wr, l_el, l_sh) > ARM_STRAIGHT_THRESH
        r_arm_straight = calculate_angle(r_wr, r_el, r_sh) > ARM_STRAIGHT_THRESH
        l_leg_straight = calculate_angle(l_hp, l_kn, l_an) > LEG_STRAIGHT_THRESH
        r_leg_straight = calculate_angle(r_hp, r_kn, r_an) > LEG_STRAIGHT_THRESH

        # --- POSITION CHECK ---
        # Arms Angle (Shoulder as pivot)
        avg_arm_lift = (calculate_angle(l_el, l_sh, l_hp) + calculate_angle(r_el, r_sh, r_hp)) / 2
        # Legs Spread (Hip as pivot relative to Shoulder)
        avg_leg_spread = (calculate_angle(l_sh, l_hp, l_kn) + calculate_angle(r_sh, r_hp, r_kn)) / 2

        # --- COUNTING LOGIC ---
        if l_arm_straight and r_arm_straight and l_leg_straight and r_leg_straight:
            
            # UP Position: Arms High AND Legs Wide
            if avg_arm_lift > ARM_UP_ANGLE and avg_leg_spread < LEG_SPREAD_UP:
                if stage == "down":
                    stage = "up"
                    counter += 1
                    feedback = "GOOD REP"
            
            # DOWN Position: Arms Low AND Legs Together
            elif avg_arm_lift < ARM_DOWN_ANGLE and avg_leg_spread > LEG_SPREAD_DOWN:
                stage = "down"
                feedback = "Ready"
        else:
            feedback = "FIX FORM (Straighten Limbs)"

        # --- MANUAL DRAWING ---
        h, w, _ = image.shape
        
        # Draw Dots
        for landmark in lm:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 5, (245, 117, 66), -1)
            
        # Draw Lines (Skeleton)
        connections = [
            (11,13), (13,15), (12,14), (14,16), # Arms
            (11,12), (11,23), (12,24), (23,24), # Torso
            (23,25), (25,27), (24,26), (26,28)  # Legs
        ]
        for start_idx, end_idx in connections:
            start = lm[start_idx]
            end = lm[end_idx]
            cv2.line(image, 
                     (int(start.x*w), int(start.y*h)), 
                     (int(end.x*w), int(end.y*h)), 
                     (255, 255, 255), 2)

    # --- UI DISPLAY ---
    # Draw Background Box
    cv2.rectangle(image, (0,0), (350,160), (245,117,16), -1)
    
    # Text Information
    cv2.putText(image, f'COUNT: {counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(image, f'STAGE: {stage}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Feedback Color (Red if error, White if good)
    fb_color = (0, 0, 255) if "FIX" in feedback else (255, 255, 255)
    cv2.putText(image, feedback, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fb_color, 2)

    cv2.imshow('Jumping Jack Counter', image)
    
    # Quit with 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
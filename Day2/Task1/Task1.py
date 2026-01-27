import cv2
import numpy as np
import os
import glob
import time

# ==========================================
# 1. CONFIGURATION
# ==========================================
CAM_FOLDERS = [
    'Day2/data/data/view1', 
    'Day2/data/data/view5', 
    'Day2/data/data/view6'
]

OUTPUT_FOLDER = 'output_canvas'
CANVAS_SIZE = (1200, 1600)  # Height, Width
MIN_CONTOUR_AREA = 500      # Ignore small noise
DELAY_MS = 200              # Delay between frames

# Adaptive Padding
PADDING_RATIO = 0.3         # 0.3 = Add 30% spacing around the group of objects

# Morphological Kernel (Helps connect gaps before we calculate the box)
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 30))

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def fit_on_canvas(canvas, rois):
    """Arranges images on the canvas."""
    h_canvas, w_canvas, _ = canvas.shape
    current_x, current_y = 0, 0
    max_row_h = 0

    for roi in rois:
        h, w, _ = roi.shape
        if w > w_canvas: continue 

        if current_x + w > w_canvas:
            current_x = 0
            current_y += max_row_h
            max_row_h = 0 

        if current_y + h > h_canvas:
            break

        canvas[current_y:current_y+h, current_x:current_x+w] = roi
        current_x += w
        max_row_h = max(max_row_h, h)

    return canvas

# ==========================================
# 3. INITIALIZATION
# ==========================================
print("Loading images...")
image_paths = [sorted(glob.glob(os.path.join(folder, '*.jpg'))) for folder in CAM_FOLDERS]

if any(len(p) == 0 for p in image_paths):
    print("Error: One or more folders are empty.")
    exit()

min_frames = min([len(p) for p in image_paths])
print(f"Processing {min_frames} synchronized frames...")

subtractors = [cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True) for _ in range(3)]

prev_frame_time = 0

# ==========================================
# 4. MAIN LOOP
# ==========================================
for i in range(min_frames):
    extracted_rois = []
    
    # --- Process each camera ---
    for cam_idx in range(3):
        path = image_paths[cam_idx][i]
        frame = cv2.imread(path)
        if frame is None: continue
        
        frame_h, frame_w, _ = frame.shape
        
        # A. Motion Detection
        mask = subtractors[cam_idx].apply(frame)
        _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
        
        # B. Find Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # --- NEW LOGIC: GROUPING ---
        # Instead of cropping immediately, we collect all valid bounding boxes first
        valid_boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                valid_boxes.append((x, y, w, h))
        
        # If we found ANY motion in this camera view, merge it into ONE crop
        if valid_boxes:
            # 1. Find the "Union" rectangle that covers ALL boxes
            min_x = min([b[0] for b in valid_boxes])
            min_y = min([b[1] for b in valid_boxes])
            max_x = max([b[0] + b[2] for b in valid_boxes])
            max_y = max([b[1] + b[3] for b in valid_boxes])
            
            # Calculate width and height of this "Union Group"
            group_w = max_x - min_x
            group_h = max_y - min_y
            
            # 2. Apply Adaptive Padding to the WHOLE Group
            pad_w = int(group_w * PADDING_RATIO)
            pad_h = int(group_h * PADDING_RATIO)

            x_new = max(0, min_x - pad_w)
            y_new = max(0, min_y - pad_h)
            x_end = min(frame_w, max_x + pad_w)
            y_end = min(frame_h, max_y + pad_h)
            
            # 3. Crop once
            roi = frame[y_new:y_end, x_new:x_end]
            extracted_rois.append(roi)

    # --- Create Canvas ---
    canvas = np.zeros((CANVAS_SIZE[0], CANVAS_SIZE[1], 3), dtype=np.uint8)
    canvas = fit_on_canvas(canvas, extracted_rois)

    # --- FPS Counter ---
    new_frame_time = time.time()
    if prev_frame_time != 0:
        fps = 1 / (new_frame_time - prev_frame_time)
        fps_text = f"FPS: {int(fps)}"
    else:
        fps_text = "FPS: 0"
    prev_frame_time = new_frame_time

    cv2.putText(canvas, fps_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    output_path = os.path.join(OUTPUT_FOLDER, f"canvas_{i:03d}.jpg")
    cv2.imwrite(output_path, canvas)
    
    cv2.imshow("Multi-Camera Canvas Stream", canvas)
    
    if cv2.waitKey(DELAY_MS) & 0xFF == 27:
        break

cv2.destroyAllWindows()
print("Processing complete.")
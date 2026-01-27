import cv2
import numpy as np
import os
import glob
import time  # <--- NEW: Needed for FPS calculation

# ==========================================
# 1. CONFIGURATION
# ==========================================
CAM_FOLDERS = [
    'Day2/data/data/view1', 
    'Day2/data/data/view5', 
    'Day2/data/data/view6'
]

OUTPUT_FOLDER = 'output_canvas'
CANVAS_SIZE = (600, 800) 
MIN_CONTOUR_AREA = 500    
PADDING = 100              
DELAY_MS = 200             # <--- set to 30ms for ~30 FPS (if processing is fast enough)

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

# --- NEW: FPS Variables ---
prev_frame_time = 0
new_frame_time = 0

# ==========================================
# 4. MAIN LOOP
# ==========================================
for i in range(min_frames):
    extracted_rois = []
    
    # Process cameras
    for cam_idx in range(3):
        path = image_paths[cam_idx][i]
        frame = cv2.imread(path)
        if frame is None: continue
        
        frame_h, frame_w, _ = frame.shape
        
        # Motion detection
        mask = subtractors[cam_idx].apply(frame)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Padding logic
                x_new = max(0, x - PADDING)
                y_new = max(0, y - PADDING)
                x_end = min(frame_w, x + w + PADDING)
                y_end = min(frame_h, y + h + PADDING)
                
                roi = frame[y_new:y_end, x_new:x_end]
                extracted_rois.append(roi)

    # Create Canvas
    canvas = np.zeros((CANVAS_SIZE[0], CANVAS_SIZE[1], 3), dtype=np.uint8)
    canvas = fit_on_canvas(canvas, extracted_rois)

    # --- NEW: FPS Calculation & Drawing ---
    new_frame_time = time.time()
    
    # Avoid division by zero on the very first frame
    if prev_frame_time != 0:
        fps = 1 / (new_frame_time - prev_frame_time)
        fps_text = f"FPS: {int(fps)}"
    else:
        fps_text = "FPS: 0"
        
    prev_frame_time = new_frame_time

    # Draw FPS on top-left of canvas (Green color)
    cv2.putText(canvas, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Output & Display
    output_path = os.path.join(OUTPUT_FOLDER, f"canvas_{i:03d}.jpg")
    cv2.imwrite(output_path, canvas)
    
    cv2.imshow("Multi-Camera Canvas Stream", canvas)
    
    if cv2.waitKey(DELAY_MS) & 0xFF == 27:
        break

cv2.destroyAllWindows()
print("Processing complete.")
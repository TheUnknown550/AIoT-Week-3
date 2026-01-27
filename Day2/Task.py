import cv2
import numpy as np
import os
import glob

# ==========================================
# 1. CONFIGURATION (Edit this section)
# ==========================================
# Define your input folders (Make sure these paths are correct!)
CAM_FOLDERS = [
    'Day2/data/data/view1', 
    'Day2/data/data/view5', 
    'Day2/data/data/view6'
]

# Output settings
OUTPUT_FOLDER = 'output_canvas'
CANVAS_SIZE = (600, 800)  # (Height, Width) of the black background
MIN_CONTOUR_AREA = 500    # Filter out small noise (leaves, wind)

# New features you requested
PADDING = 40              # Pixels to expand the box (makes images bigger)
DELAY_MS = 200            # Delay between frames in milliseconds (0 = wait for keypress)

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def fit_on_canvas(canvas, rois):
    """
    Arranges a list of images (ROIs) onto the canvas 
    using a simple "left-to-right, then new row" strategy.
    """
    h_canvas, w_canvas, _ = canvas.shape
    current_x, current_y = 0, 0
    max_row_h = 0

    for roi in rois:
        h, w, _ = roi.shape
        
        # Skip if image is wider than the canvas itself
        if w > w_canvas: 
            continue 

        # Check if we need to wrap to a new row (if image goes off right edge)
        if current_x + w > w_canvas:
            current_x = 0
            current_y += max_row_h
            max_row_h = 0  # Reset height tracker for new row

        # Stop if we run out of vertical space
        if current_y + h > h_canvas:
            break

        # Place the image on the canvas
        canvas[current_y:current_y+h, current_x:current_x+w] = roi
        
        # Update cursor position
        current_x += w
        max_row_h = max(max_row_h, h)

    return canvas

# ==========================================
# 3. INITIALIZATION
# ==========================================
# Load file paths and sort them to ensure synchronization
print("Loading images...")
image_paths = [sorted(glob.glob(os.path.join(folder, '*.jpg'))) for folder in CAM_FOLDERS]

# Safety check: Ensure all folders have images
if any(len(p) == 0 for p in image_paths):
    print("Error: One or more folders are empty. Check your paths in 'CAM_FOLDERS'.")
    exit()

# Find the minimum number of frames (so we don't crash if one camera has fewer)
min_frames = min([len(p) for p in image_paths])
print(f"Processing {min_frames} synchronized frames from {len(CAM_FOLDERS)} cameras...")

# Initialize Background Subtractors (MOG2 algorithm)
# history=500: Learns background over time
# varThreshold=50: Sensitivity (lower = detects more motion)
subtractors = [cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True) for _ in range(3)]

# ==========================================
# 4. MAIN PROCESSING LOOP
# ==========================================
for i in range(min_frames):
    extracted_rois = []
    
    # --- Process each camera for the current time step ---
    for cam_idx in range(3):
        # Read the frame
        path = image_paths[cam_idx][i]
        frame = cv2.imread(path)
        
        if frame is None:
            continue
        
        frame_h, frame_w, _ = frame.shape
        
        # A. Motion Detection (Get the mask)
        mask = subtractors[cam_idx].apply(frame)
        
        # B. Noise Cleaning (Remove shadows and small white dots)
        # Threshold: removes grey shadows (values < 254 become 0)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        # Morphology: removes "salt and pepper" noise
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # C. Find Contours (The moving objects)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Filter small movements
            if area > MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # --- APPLY PADDING (With Boundary Safety) ---
                # We expand the box by 'PADDING', but clamp to 0 and frame size 
                # to avoid errors if the object is at the edge of the screen.
                x_new = max(0, x - PADDING)
                y_new = max(0, y - PADDING)
                x_end = min(frame_w, x + w + PADDING)
                y_end = min(frame_h, y + h + PADDING)
                
                # Cut out the image (ROI)
                roi = frame[y_new:y_end, x_new:x_end]
                
                # (Optional) Draw a box on the ROI to show detection
                # cv2.rectangle(roi, (0,0), (x_end-x_new, y_end-y_new), (0, 255, 0), 2)
                
                extracted_rois.append(roi)

    # --- Create the Final Canvas ---
    # Start with a black image
    canvas = np.zeros((CANVAS_SIZE[0], CANVAS_SIZE[1], 3), dtype=np.uint8)
    
    # Pack our extracted images onto it
    canvas = fit_on_canvas(canvas, extracted_rois)

    # --- Output & Display ---
    # Save to file
    output_path = os.path.join(OUTPUT_FOLDER, f"canvas_{i:03d}.jpg")
    cv2.imwrite(output_path, canvas)
    
    # Show on screen
    cv2.imshow("Multi-Camera Canvas Stream", canvas)
    
    # Wait for delay (Press ESC to quit early)
    if cv2.waitKey(DELAY_MS) & 0xFF == 27:
        print("User aborted.")
        break

# Cleanup
cv2.destroyAllWindows()
print("Processing complete. Saved images to:", OUTPUT_FOLDER)
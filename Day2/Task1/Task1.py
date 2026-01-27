import cv2
import numpy as np
import os
import glob
import time
import psutil
import json
from collections import defaultdict

# ==========================================
# 1. CONFIGURATION
# ==========================================
CAM_FOLDERS = [
    'data/data/view1', 
    'data/data/view5', 
    'data/data/view6'
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
# EVALUATION METRICS TRACKING
# ==========================================
metrics = {
    'fps_values': [],
    'latency_ms': [],
    'detection_counts': [],
    'roi_sizes': [],
    'processing_time': {
        'motion_detection': [],
        'contour_finding': [],
        'roi_extraction': [],
        'canvas_creation': [],
        'total': []
    },
    'memory_usage_mb': [],
    'frames_processed': 0
}

process = psutil.Process(os.getpid())

# ==========================================
# 4. MAIN LOOP
# ==========================================
for i in range(min_frames):
    frame_start_time = time.time()
    extracted_rois = []
    detection_count = 0
    
    # Timing for different stages
    motion_time = 0
    contour_time = 0
    extraction_time = 0
    
    # --- Process each camera ---
    for cam_idx in range(3):
        path = image_paths[cam_idx][i]
        frame = cv2.imread(path)
        if frame is None: continue
        
        frame_h, frame_w, _ = frame.shape
        
        # A. Motion Detection
        t1 = time.time()
        mask = subtractors[cam_idx].apply(frame)
        _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
        motion_time += (time.time() - t1)
        
        # B. Find Contours
        t2 = time.time()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_time += (time.time() - t2)
        
        # --- NEW LOGIC: GROUPING ---
        t3 = time.time()
        # Instead of cropping immediately, we collect all valid bounding boxes first
        valid_boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                valid_boxes.append((x, y, w, h))
                detection_count += 1
        
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
            metrics['roi_sizes'].append(roi.shape[0] * roi.shape[1])  # Track ROI size
        
        extraction_time += (time.time() - t3)

    # --- Create Canvas ---
    canvas_start = time.time()
    canvas = np.zeros((CANVAS_SIZE[0], CANVAS_SIZE[1], 3), dtype=np.uint8)
    canvas = fit_on_canvas(canvas, extracted_rois)
    canvas_time = time.time() - canvas_start

    # --- Calculate Metrics ---
    frame_end_time = time.time()
    frame_latency = (frame_end_time - frame_start_time) * 1000  # in ms
    
    # FPS Counter
    new_frame_time = time.time()
    if prev_frame_time != 0:
        fps = 1 / (new_frame_time - prev_frame_time)
        fps_text = f"FPS: {int(fps)}"
        metrics['fps_values'].append(fps)
    else:
        fps_text = "FPS: 0"
    prev_frame_time = new_frame_time
    
    # Store metrics
    metrics['latency_ms'].append(frame_latency)
    metrics['detection_counts'].append(detection_count)
    metrics['processing_time']['motion_detection'].append(motion_time * 1000)
    metrics['processing_time']['contour_finding'].append(contour_time * 1000)
    metrics['processing_time']['roi_extraction'].append(extraction_time * 1000)
    metrics['processing_time']['canvas_creation'].append(canvas_time * 1000)
    metrics['processing_time']['total'].append(frame_latency)
    metrics['memory_usage_mb'].append(process.memory_info().rss / 1024 / 1024)
    metrics['frames_processed'] = i + 1
    
    # Display metrics on canvas
    cv2.putText(canvas, fps_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(canvas, f"Latency: {frame_latency:.1f}ms", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(canvas, f"Detections: {detection_count}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(canvas, f"ROIs: {len(extracted_rois)}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 255), 2)

    output_path = os.path.join(OUTPUT_FOLDER, f"canvas_{i:03d}.jpg")
    cv2.imwrite(output_path, canvas)
    
    cv2.imshow("Multi-Camera Canvas Stream", canvas)
    
    if cv2.waitKey(DELAY_MS) & 0xFF == 27:
        break

cv2.destroyAllWindows()

# ==========================================
# FINAL METRICS REPORT
# ==========================================
print("\n" + "="*60)
print("PERFORMANCE METRICS SUMMARY")
print("="*60)

# Calculate statistics
avg_fps = np.mean(metrics['fps_values']) if metrics['fps_values'] else 0
avg_latency = np.mean(metrics['latency_ms'])
min_latency = np.min(metrics['latency_ms'])
max_latency = np.max(metrics['latency_ms'])
std_latency = np.std(metrics['latency_ms'])

avg_detections = np.mean(metrics['detection_counts'])
total_detections = np.sum(metrics['detection_counts'])

avg_roi_size = np.mean(metrics['roi_sizes']) if metrics['roi_sizes'] else 0
avg_memory = np.mean(metrics['memory_usage_mb'])

print(f"\n📊 THROUGHPUT METRICS:")
print(f"  • Average FPS: {avg_fps:.2f}")
print(f"  • Total Frames Processed: {metrics['frames_processed']}")
print(f"  • Total Processing Time: {sum(metrics['processing_time']['total'])/1000:.2f}s")

print(f"\n⏱️  LATENCY METRICS:")
print(f"  • Average Latency: {avg_latency:.2f} ms")
print(f"  • Min Latency: {min_latency:.2f} ms")
print(f"  • Max Latency: {max_latency:.2f} ms")
print(f"  • Std Deviation: {std_latency:.2f} ms")

print(f"\n🎯 DETECTION METRICS:")
print(f"  • Total Detections: {total_detections}")
print(f"  • Avg Detections/Frame: {avg_detections:.2f}")
print(f"  • Avg ROI Size: {avg_roi_size:.0f} pixels²")

print(f"\n⚙️  PROCESSING TIME BREAKDOWN (avg per frame):")
print(f"  • Motion Detection: {np.mean(metrics['processing_time']['motion_detection']):.2f} ms")
print(f"  • Contour Finding: {np.mean(metrics['processing_time']['contour_finding']):.2f} ms")
print(f"  • ROI Extraction: {np.mean(metrics['processing_time']['roi_extraction']):.2f} ms")
print(f"  • Canvas Creation: {np.mean(metrics['processing_time']['canvas_creation']):.2f} ms")
print(f"  • Total: {avg_latency:.2f} ms")

print(f"\n💾 RESOURCE USAGE:")
print(f"  • Average Memory: {avg_memory:.2f} MB")
print(f"  • Peak Memory: {max(metrics['memory_usage_mb']):.2f} MB")

# Calculate throughput
throughput = metrics['frames_processed'] / (sum(metrics['processing_time']['total']) / 1000)
print(f"\n🚀 SYSTEM THROUGHPUT:")
print(f"  • Frames/Second: {throughput:.2f} FPS")
print(f"  • Detections/Second: {total_detections / (sum(metrics['processing_time']['total']) / 1000):.2f}")

# Save metrics to JSON
metrics_file = os.path.join(OUTPUT_FOLDER, 'performance_metrics.json')
with open(metrics_file, 'w') as f:
    # Convert numpy values to Python native types for JSON serialization
    json_metrics = {
        'summary': {
            'avg_fps': float(avg_fps),
            'avg_latency_ms': float(avg_latency),
            'min_latency_ms': float(min_latency),
            'max_latency_ms': float(max_latency),
            'std_latency_ms': float(std_latency),
            'total_detections': int(total_detections),
            'avg_detections_per_frame': float(avg_detections),
            'avg_roi_size_pixels': float(avg_roi_size),
            'avg_memory_mb': float(avg_memory),
            'peak_memory_mb': float(max(metrics['memory_usage_mb'])),
            'throughput_fps': float(throughput),
            'frames_processed': metrics['frames_processed']
        },
        'processing_time_breakdown_ms': {
            'motion_detection': float(np.mean(metrics['processing_time']['motion_detection'])),
            'contour_finding': float(np.mean(metrics['processing_time']['contour_finding'])),
            'roi_extraction': float(np.mean(metrics['processing_time']['roi_extraction'])),
            'canvas_creation': float(np.mean(metrics['processing_time']['canvas_creation']))
        }
    }
    json.dump(json_metrics, f, indent=2)

print(f"\n📁 Metrics saved to: {metrics_file}")
print("="*60)
print("Processing complete.")

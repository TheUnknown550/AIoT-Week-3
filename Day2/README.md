# Multi-Camera Motion Detection and Canvas Display

## Overview
This project processes multiple camera views to detect motion, extract regions of interest (ROIs), and display them on a unified canvas in real-time. The system uses background subtraction for motion detection and arranges detected objects from multiple camera feeds onto a single output display.

## Features
- **Multi-Camera Support**: Processes 3 synchronized camera views (view1, view5, view6)
- **Motion Detection**: Uses MOG2 background subtraction algorithm
- **ROI Extraction**: Detects and extracts moving objects with padding
- **Canvas Display**: Arranges multiple ROIs on a single canvas
- **FPS Display**: Shows real-time processing speed
- **Frame Export**: Saves each processed frame to disk

## Requirements
- Python 3.x
- OpenCV (cv2)
- NumPy

Install dependencies:
```bash
pip install opencv-python numpy
```

## Directory Structure
```
Day2/
├── Task.py              # Main script
├── data/
│   └── data/
│       ├── view1/       # Camera 1 frames (*.jpg)
│       ├── view5/       # Camera 5 frames (*.jpg)
│       └── view6/       # Camera 6 frames (*.jpg)
└── output_canvas/       # Generated canvas frames
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAM_FOLDERS` | view1, view5, view6 | Paths to camera image folders |
| `OUTPUT_FOLDER` | output_canvas | Output directory for canvas frames |
| `CANVAS_SIZE` | (600, 800) | Canvas dimensions (height, width) |
| `MIN_CONTOUR_AREA` | 500 | Minimum contour area for detection |
| `PADDING` | 100 | Pixels to add around detected objects |
| `DELAY_MS` | 200 | Display delay in milliseconds |

### Background Subtractor Settings
- **History**: 500 frames
- **Variance Threshold**: 50
- **Detect Shadows**: True

## How It Works

1. **Initialization**
   - Loads image paths from all camera folders
   - Synchronizes frames across cameras
   - Creates background subtractor for each camera

2. **Processing Loop** (for each frame):
   - **Motion Detection**: Apply MOG2 background subtraction
   - **Mask Processing**: Threshold and morphological operations
   - **Contour Detection**: Find external contours
   - **ROI Extraction**: Extract bounding boxes with padding for significant contours
   - **Canvas Creation**: Arrange all ROIs on a black canvas
   - **FPS Calculation**: Compute and display processing speed
   - **Output**: Save frame and display window

3. **Canvas Layout**
   - ROIs are arranged left-to-right, top-to-bottom
   - Automatic row wrapping when width is exceeded
   - Stops adding ROIs when canvas height is full

## Usage

1. Ensure your camera frames are in the correct folders:
   - `data/data/view1/`
   - `data/data/view5/`
   - `data/data/view6/`

2. Run the script from the Day2 directory:
   ```bash
   cd Day2
   python Task.py
   ```

3. View the output:
   - Real-time display window: "Multi-Camera Canvas Stream"
   - Saved frames: `output_canvas/canvas_XXX.jpg`
   - Press ESC to exit early

## Output
- **Display Window**: Shows the canvas with FPS counter (green text, top-left)
- **Saved Frames**: Sequential images in `output_canvas/` folder
  - Format: `canvas_000.jpg`, `canvas_001.jpg`, etc.

## Algorithm Details

### Motion Detection Pipeline
1. Apply MOG2 background subtraction
2. Threshold mask to binary (removes shadows)
3. Apply morphological opening (5x5 kernel) to reduce noise
4. Find contours in processed mask
5. Filter contours by minimum area (500 pixels)
6. Extract ROIs with padding (100 pixels on each side)

### Canvas Arrangement
- Greedy left-to-right, top-to-bottom packing
- Skips oversized images (wider than canvas)
- Maintains aspect ratio of all ROIs
- No resizing of extracted regions

## Troubleshooting

**Error: "One or more folders are empty"**
- Ensure all three camera folders contain .jpg images
- Verify folder paths are correct relative to execution directory

**Low FPS**
- Reduce DELAY_MS for faster playback
- Decrease canvas size
- Reduce number of cameras

**Missing detections**
- Lower MIN_CONTOUR_AREA threshold
- Adjust MOG2 varThreshold parameter
- Increase PADDING for larger ROIs

## Notes
- All camera views must have the same number of frames
- System processes only up to the minimum frame count
- Background subtraction requires initial frames for learning
- First few frames may have incomplete motion detection

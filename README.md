# AIoT Week 3 - Multi-Camera Motion Detection

## Project Overview
A multi-camera system that detects motion from 3 synchronized camera views and displays detected objects on a unified canvas.

## What We Did

### Day 1
- **Task 1**: MobileNetV2 implementation
- **Task 2**: Client-server image processing with socket communication

### Day 2
Multi-camera motion detection and canvas display system:
- Processes 3 camera views (view1, view5, view6)
- Detects motion using background subtraction
- Extracts moving objects and displays them on one canvas
- Shows real-time FPS counter

## Quick Start

### Day 2 - Motion Detection
```bash
cd Day2
python Task.py
```

Press ESC to exit.

## Key Features
- **Background Subtraction**: MOG2 algorithm for motion detection
- **Multi-Camera Sync**: Processes frames from 3 cameras simultaneously
- **Smart Canvas Layout**: Automatically arranges detected objects
- **Real-time Display**: Shows FPS and live detection results

## Configuration (Day2/Task.py)
- **Canvas Size**: 600x800 pixels
- **Min Detection Area**: 500 pixels
- **Padding**: 100 pixels around detected objects
- **Display Delay**: 200ms between frames

## Output
- Live preview window with FPS counter
- Saved frames in `Day2/output_canvas/` folder

## Requirements
```bash
pip install opencv-python numpy tensorflow
```

## Troubleshooting
- **"Folders are empty" error**: Run script from `Day2` directory
- **No detections**: Lower MIN_CONTOUR_AREA or adjust lighting
- **Slow processing**: Reduce canvas size or increase DELAY_MS

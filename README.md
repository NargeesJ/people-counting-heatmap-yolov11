# People Counting and Heatmap Visualization using YOLO + Ultralytics

## Project Overview

This project performs real-time human detection, tracking IN/OUT counting and heatmap visualization using YOLO (You Only Look Once) and Ultralytics' Heatmap solution. It analyzes movement patterns in a given video and visually identifies areas of high activity using a dynamic heatmap overlay.

## Detection Method

### 1. YOLO Model

The detection is performed using the YOLOv11n model (yolo11n.pt), which is a lightweight and fast object detector trained on the COCO dataset. It detects multiple object categories, but in this script, only person detection is used.

```python
model = YOLO('yolo11n.pt')
PERSON_CLASS_ID = [0]  # COCO ID for 'person'
```

### 2. Tracking

To maintain identity across frames, BoT-SORT tracker (tracker='botsort.yaml') is used. This ensures that each detected person receives a unique ID throughout the video, allowing reliable IN/OUT counting.

```python
results = model.track(frame, persist=True, tracker='botsort.yaml', verbose=False)[0]
```

## Line Coordinates and Scaling

Two parallel horizontal lines are defined across the frame — one for IN counting and one for OUT counting. Their coordinates are based on a reference frame resolution of 1127×625, and automatically scaled to match the input video size.

| Line | Start (x, y) | End (x, y) | Description |
|------|--------------|------------|-------------|
| IN Line | (2, 529) | (1125, 516) | When a person crosses this line from top → bottom, they are counted as 'IN'. |
| OUT Line | (0, 570) | (1125, 557) | When a person crosses this line from bottom → top, they are counted as 'OUT'. |

These are scaled dynamically for the input video to maintain correct proportional positions even if the video has a different resolution.

## IN/OUT Counting Logic Explanation

The IN/OUT counting system analysis using tracked center points. Each tracked person has a center point calculated per frame, and their recent positions are stored. When the trajectory crosses a counting line, an action is checked.

### IN Count Logic

If the person moves from above to below the IN line, they are counted as 'IN':

```python
if prev_center[1] < LINE_IN_START[1] and curr_center[1]:
    in_count += 1
```

### OUT Count Logic

If the person moves from below to above the OUT line, they are counted as 'OUT':

```python
if prev_center[1] > LINE_OUT_START[1] == curr_center[1]:
    out_count += 1
```

Each track_id is counted only once per direction to prevent duplicate counting if the same person hovers near a line.

## Heatmap Visualization [Ultralytics Solution]

For visual density-based analysis, the project uses the Ultralytics Heatmap solution. It creates a smooth density-based heatmap where each detection adds a point, and the final blended output is generated automatically.

```python
from ultralytics.solutions.heatmap import Heatmap
heatmap.add_point(center)
frame = heatmap.generate_heatmap(frame)
```

The heatmap becomes more intense in regions where people stay or move frequently, helping visualize foot traffic density.

## Output

The script saves two files:
- Processed video with detections, lines, counts, and heatmap overlay → output.mp4
- Final frame image snapshot → last_frame.jpg

Example overlay includes:
- Green circles for detected centers
- Blue "IN" line
- Red "OUT" line
- Heatmap overlay showing movement intensity
- "In: X | Out: Y" counters displayed in the top-left corner

## Summary of Process Flow

1. Load YOLO model to detect persons in each frame.
2. Track each person using BoT-SORT to maintain consistent IDs.
3. Compute movement direction based on previous and current center points.
4. Check line crossings and update IN/OUT counters.
5. Accumulate movement density and render a real-time heatmap overlay.
6. Export annotated video and last frame snapshot.
from ultralytics import YOLO
import cv2
import numpy as np
import json
#load yolo
model = YOLO('yolov8n.pt')

#define vehicle classes


#load video
cap = cv2.VideoCapture('./video_01.mp4')
ret, frame = cap.read()
if not ret:
    raise ValueError("Failed to read video")

height, width, channels = frame.shape
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('./results/yolo_annotated_video_01.mp4', fourcc, fps, (width, height))

bg = cv2.createBackgroundSubtractorMOG2(
    history = 50, varThreshold = 50, detectShadows = False)



# Define vehicle classes and visualization properties
vehicle_classes = {
    'car': {'color': (0, 255, 0), 'radius': 4},    # green
    'truck': {'color': (0, 0, 255), 'radius': 6},  # red
    'bus': {'color': (255, 0, 0), 'radius': 8}     # blue
}


# 4) Process video frame by frame and save jason file for vehicle locations
all_frames_boxes = []  # list to store info for each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    bgmask = bg.apply(frame)
    bgmask = cv2.medianBlur(bgmask,5)
    
    # Run YOLOv8 prediction
    results = model.predict(
        source=frame,      # input frame
        imgsz=640,         # resize for model
        conf=0.25,         # confidence threshold
        iou=0.45,          # NMS IoU threshold
        device='cpu',      # or 'cuda:0' for GPU
        stream=False
    )

    # Get the first result (single frame)
    res = results[0]
    frame_boxes = []
    
    # Draw boxes for detected vehicles
    if hasattr(res, 'boxes') and len(res.boxes):
        for box in res.boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names.get(class_id, str(class_id))

            # Only draw vehicle classes
            if class_name in vehicle_classes:
                motion_region = bgmask[y1:y2,x1:x2]
                motion_ratio = np.mean(motion_region > 0)
                if motion_ratio < 0.03:
                    continue

                color = vehicle_classes[class_name]['color']
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Add to 2D plane
                radius = vehicle_classes[class_name]['radius']
                x_center = int((x1 + x2) / 2)
                y_bottom = int(y2)

                frame_boxes.append({
                    'class': class_name,
                    'conf': confidence,
                    'bbox': [x1, y1, x2, y2],
                    'center': [x_center, y_bottom]
                })

    all_frames_boxes.append(frame_boxes)
    output_video.write(frame)

# Release resources
cap.release()
output_video.release()

# Save to JSON

with open('./results/vehicle_bboxes.json', 'w') as f:
    json.dump(all_frames_boxes, f)

# Load the BEV image
bev_image = cv2.imread('2020-25-06.png')  # or .jpg
plane_image = bev_image.copy()  # reset per frame
# Manutally find pixel locations from the camera image and the related BEV image using software ImageJ

camera_coordiate = np.array([[551,1032],[684,926],[872,766],[1133,503],[1105,905],[1155,835],
                            [1264,682],[1295,636],[1148,504],[1625,598],[1161,914],[1848,1020],[703,354],[727,397]])
bev_coordiante = np.array([[506,262],[527,328],[566,394],[691,605],[456,416],[473,447],
                          [513,535],[532,572],[683,610],[449,715],[320,541],[244,544],[1168,544],[1044,520]])
H, status = cv2.findHomography(camera_coordiate, bev_coordiante)

#change bounding box locations to bottom-center and map coordinates to BEV image
bev_coords_all_frames = []

for frame_boxes in all_frames_boxes:
    frame_bev = []
    for box in frame_boxes:
        x1, y1, x2, y2 = box['bbox']
        x_center = (x1 + x2) / 2
        y_bottom = y2
        vehicle_pixel = np.array([x_center, y_bottom, 1])
        
        # Map to BEV
        vehicle_bev = np.dot(H, vehicle_pixel)
        vehicle_bev /= vehicle_bev[2]  # normalize

        frame_bev.append({
            'class': box['class'],
            'conf': box['conf'],
            'bev_x': vehicle_bev[0],
            'bev_y': vehicle_bev[1]
        })
    bev_coords_all_frames.append(frame_bev)

with open('./results/vehicle_bev_coords.json', 'w') as f:
    json.dump(bev_coords_all_frames, f)



frame_bev = bev_coords_all_frames[0]  # list of dicts with 'bev_x', 'bev_y'
for v in frame_bev:
    x_px = int(v['bev_x'])
    y_px = int(v['bev_y'])
    cls = v['class']
    color = vehicle_classes.get(cls, {'color': (0,0,255)})['color']
    radius = vehicle_classes.get(cls, {'radius': 5})['radius']
    cv2.circle(plane_image, (x_px, y_px), radius, color, -1)
    cv2.putText(plane_image, cls, (x_px-20, y_px-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
# Save the 2D plane visualization
cv2.imwrite('./results/bev_with_vehicles.png', plane_image)

import cv2
import numpy as np
from ultralytics import YOLO
import torch

class cv_Yolo:

    #def __init__(self, yolo_model='yolov8n.pt', confidence=0.5, threshold=0.3):
    def __init__(self, yolo_model='yolov8l.pt', confidence=0.5, threshold=0.3):    
        self.confidence = confidence
        self.threshold = threshold

        # Load the YOLOv8 model
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device='cuda'
        self.model = YOLO(yolo_model).to(self.device)
        self.model.conf = confidence  # NMS confidence threshold
        self.model.iou = threshold  # NMS IoU threshold

    def detect(self, image):
        # Convert the image to RGB (if it's not already in that format)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run inference
        results = self.model(image_rgb, device=self.device)

        detections = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, score, class_id in zip(boxes, scores, class_ids):
                if score >= self.confidence:
                    top_left = (int(box[0]), int(box[1]))
                    bottom_right = (int(box[2]), int(box[3]))
                    box_2d = [top_left, bottom_right]
                    class_ = self.get_class(class_id)
                    if class_ == "person":
                        class_ = "pedestrian"
                    detections.append(Detection(box_2d, class_))

        return detections

    def get_class(self, class_id):
        # Use the model's class names
        return self.model.names[class_id]

class Detection:
    def __init__(self, box_2d, class_):
        self.box_2d = box_2d
        self.detected_class = class_

# Usage example
if __name__ == "__main__":
    yolo = cv_Yolo(yolo_model='yolov8n.pt', confidence=0.5, threshold=0.3)
    image = cv2.imread('/home/user1/Desktop/Pirooz testing/20240310_183713.jpg')  # Provide path to your image
    detections = yolo.detect(image)
    for detection in detections:
        print(f"Detected {detection.detected_class} at {detection.box_2d}")

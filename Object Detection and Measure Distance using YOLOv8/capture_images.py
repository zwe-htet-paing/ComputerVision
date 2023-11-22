import os
import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

import supervision as sv


class ObjectDetection:

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)

    def load_model(self):
        model = YOLO("yolov8m.pt")  # load a pretrained YOLOv8n model
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame)
        return results

    def plot_bboxes(self, results, frame, target_class="person"):
        xyxys = []
        confidences = []
        class_ids = []

        # Extract detections for the specified class
        for result in results:
            boxes = result.boxes.cpu().numpy()

            if len(boxes) > 0:
                class_id = boxes.cls[0]
                conf = boxes.conf[0]
                xyxy = boxes.xyxy[0]

                if self.CLASS_NAMES_DICT[class_id] == target_class:
                    xyxys.append(result.boxes.xyxy.cpu().numpy())
                    confidences.append(result.boxes.conf.cpu().numpy())
                    class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

        # Check if there are detections for the target class
        if xyxys:
            # Setup detections for visualization
            detections = sv.Detections(
                xyxy=np.concatenate(xyxys),
                confidence=np.concatenate(confidences),
                class_id=np.concatenate(class_ids),
            )

            # Format custom labels
            self.labels = [f"{target_class} {confidence:0.2f}"
                        for _, _, confidence, _, _
                        in detections]

            # Annotate and display frame
            frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)

        return frame

def create_directory(dir_name):
    # Check if the directory exists, and create it if not
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def main():
    dir_name = "ReferenceImages"
    target_class = "person"
    number_image_captured = 10
    
    cam_number = 0  # choose camera number
    cap = cv2.VideoCapture(cam_number)
    
    assert cap.isOpened()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = ObjectDetection()
    counter = 0
    capture = False
    number = 0

    while True:
        create_directory(dir_name)
        start_time = time()

        ret, frame = cap.read()
        assert ret

        original = frame.copy()
        cv2.imshow('original', original)

        results = detector.predict(frame)
        frame = detector.plot_bboxes(results, frame, target_class=target_class)

        end_time = time()
        fps = 1 / np.round(end_time - start_time, 2)

        cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        if capture and counter < number_image_captured:
            counter += 1
            cv2.putText(
                frame, f"Capturing Img No: {number}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            counter = 0

        cv2.imshow('YOLOv8 Detection', frame)

        key = cv2.waitKey(1)
        if key == ord('c'):
            capture = True
            number += 1
            cv2.imwrite(f'ReferenceImages/image{number}.png', original)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

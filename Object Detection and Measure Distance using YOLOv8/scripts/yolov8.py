import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import supervision as sv

class ObjectDetection:

    def __init__(self, capture_index):
        self.capture_index = capture_index
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


    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            start_time = time()
            ret, frame = cap.read()
            assert ret
            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame, target_class="person")
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('YOLOv8 Detection', frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ObjectDetection(capture_index=0)
    detector()

import cv2
from ultralytics import YOLO
import math

class ObjectDetector:
    def __init__(self, model_path='yolov8m.pt', class_name='person'):
        self.model = YOLO(model_path)
        self.model.fuse()
        self.class_names_dict = self.model.model.names
        self.CLASS_NAME = class_name

        # Colors for object detected
        self.COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
        self.GREEN = (0, 255, 0)
        self.BLACK = (0, 0, 0)

        # Defining fonts
        self.FONTS = cv2.FONT_HERSHEY_COMPLEX

    def detect_objects(self, image):
        results = self.model(image)
        data_list = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                color = self.COLORS[int(box.cls[0]) % len(self.COLORS)]
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = self.class_names_dict[cls]
                
                if class_name != self.CLASS_NAME:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)

                label = f'{class_name}{conf}'

                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3

                cv2.rectangle(image, (x1, y1), c2, color, -1, cv2.LINE_AA)
                cv2.putText(image, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

                if class_name == self.CLASS_NAME:
                    data_list.append([class_name, x2, (x1, y1 - 2)])

        return data_list

class MeasurementCalculator:
    @staticmethod
    def calculate_focal_length(measured_distance, real_width, width_in_frame):
        focal_length = (width_in_frame * measured_distance) / real_width
        return focal_length

    @staticmethod
    def calculate_distance(focal_length, real_object_width, width_in_frame):
        distance = (real_object_width * focal_length) / width_in_frame
        return distance

def main():
    # Distance constants
    KNOWN_DISTANCE = 45  # INCHES
    KNOWN_OBJECT_WIDTH = 16  # INCHES
    target_class = 'person'
        
    obj_detector = ObjectDetector(class_name=target_class)

    ref_person = cv2.imread('ReferenceImages/image3.png')
    person_data = obj_detector.detect_objects(ref_person)
    person_width_in_frame = person_data[0][1]
    
    calculator = MeasurementCalculator()
    focal_person = calculator.calculate_focal_length(KNOWN_DISTANCE, KNOWN_OBJECT_WIDTH,
                                                      person_width_in_frame)

    cap = cv2.VideoCapture(0)

    assert cap.isOpened()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()

        data = obj_detector.detect_objects(frame)
        print(data)

        for d in data:
            if d[0] == 'person':
                distance = calculator.calculate_distance(focal_person, KNOWN_OBJECT_WIDTH, d[1])
                x, y = d[2]
                cv2.rectangle(frame, (x, y - 3), (x + 150, y + 23), obj_detector.BLACK, -1)
                cv2.putText(frame, f'Dis: {round(distance, 2)} inch', (x + 5, y + 13), obj_detector.FONTS, 0.48,
                            obj_detector.GREEN, 2)

        cv2.imshow('frame', frame)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()

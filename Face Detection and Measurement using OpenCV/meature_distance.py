import cv2

# Constants for measurements
KNOWN_DISTANCE = 20  # in centimeters
KNOWN_FACE_WIDTH = 14.3  # centimeters

class FaceDetector:
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

    def detect(self, frame):
        face_width = 0
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray_image, 1.3, 5)

        for (x, y, h, w) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
            face_width = w

        return face_width, frame

class DistanceCalculator:
    @staticmethod
    def calculate_focal_length(measured_distance, face_real_width, face_width_in_image):
        return (face_width_in_image * measured_distance) / face_real_width

    @staticmethod
    def calculate_distance(face_real_width, focal_length, face_width_in_image):
        return (face_real_width * focal_length) / face_width_in_image

def main():
    reference_image = cv2.imread("capture_images/frame-1.png")
    
    # Instantiate FaceDetector and DistanceCalculator
    face_detector = FaceDetector()
    distance_caculator = DistanceCalculator()

    # Detect face in the reference image
    face_width, image = face_detector.detect(reference_image)
    cv2.imshow("reference_image", image)

    # Calculate focal length using reference measurements
    focal_length = distance_caculator.calculate_focal_length(KNOWN_DISTANCE, KNOWN_FACE_WIDTH, face_width)
    print("Focal Length:", focal_length)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # Detect face in the frame
        face_width, frame = face_detector.detect(frame)

        # Measure the distance using focal length and face width
        distance = distance_caculator.calculate_distance(KNOWN_FACE_WIDTH, focal_length, face_width)

        # Display the distance on the frame
        cv2.putText(frame, f" Distance = {distance}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)

        # Display the frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

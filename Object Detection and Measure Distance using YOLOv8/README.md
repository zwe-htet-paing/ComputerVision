# Real-Time Object Distance Measurement using YOLO

This project consists of two main scripts for real-time image capture and object distance measurement using YOLO and OpenCV.

## 1. capture_images.py

### Overview

This script detects and captures real-time frames of objects from a camera using YOLO and OpenCV. It allows users to capture images by pressing the 'c' key, and the images are saved in a specified directory.

### Features

1. **Real-Time Frame Capture:** Continuously captures frames from the camera feed.

2. **Image Saving:** Captures and saves images with a specific naming convention in the "ReferenceImages" directory.

3. **On-Screen Indicators:** Displays on-screen indicators, such as the current status of capturing, detected objects, and the height and width of the saving frame.

### Configuration

- **target_class**: Specify the class name of the object you want to detect.

- **number_image_captured**: Adjust the number of images to capture before stopping.

- **cam_number**: Choose the camera number (default is set to 0).


### Usage

1. Run the script:

    ```bash
    python capture_images.py
    ```

2. The camera window will open, displaying real-time frames.

3. Press the 'c' key to capture and save an image.

4. The script will create and save images in the "capture_images" directory.

5. Press 'q' or 'Esc' to exit the application.


## 2. distance_estimator.py

### Overview

This script utilizes YOLO for object detection and calculates the distance between the camera and a detected object (e.g., person) based on known measurements. The distance is calculated using the focal length, which is determined from a reference image where the object width is known.

### Features

1. **Real-Time Object Detection:** Detects objects in real-time using YOLO and OpenCV.

2. **Dynamic Distance Calculation:** Calculates the distance to the detected object dynamically as the object moves.

3. **Focal Length Calculation:** Determines the focal length using a reference image where the object width is known.

4. **On-Screen Distance Display:** Displays the calculated distance on the video feed.

5. **Adjustable Constants:** Allows configuration of constants such as `KNOWN_DISTANCE` and `KNOWN_OBJECT_WIDTH` for accurate distance calculation.

6. **Adaptable for Different Environments:** Can be adapted for various environments and scenarios where accurate object distance measurement is required.

7. **Customizable Integration:** Provides a foundation for integration into larger projects or systems requiring real-time object distance measurements.


### Configuration

- **target_class**: Specify the class name of the object you want to detect.

- **KNOWN_DISTANCE**: The known distance (in inches) between the camera and the person in the reference image.

- **KNOWN_OBJECT_WIDTH**: The known width (in inches) of the object (e.g, person) in the reference image.


### Usage

1. Run the script:
    ```bash
    python distance_estimator.py
    ```
    
2. The camera will open, and the script will detect objects in real-time. The calculated distance to the object will be displayed on the video feed.

3. Press 'q' or 'Esc' to exit the application.


## Setup

Before running the script, ensure you have the necessary dependencies installed:

```bash
pip install -r requirements.txt
```

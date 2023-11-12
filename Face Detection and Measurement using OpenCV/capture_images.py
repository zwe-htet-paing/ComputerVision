import cv2
import time
import os

def create_directory(dir_name):
    # Check if the directory exists, and create it if not
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def draw_text(image, text, position, font_scale=0.4, color=(0, 255, 0), thickness=1):
    # Draw text on the image
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_COMPLEX, font_scale, color, thickness)

def main():
    starting_time = time.time()
    frame_counter = 0
    cap_frame = 0
    dir_name = "capture_images"
    number_image_captured = 20
    capture_image = False

    cam_number = 0  # choose camera number
    camera = cv2.VideoCapture(cam_number)

    while True:
        # Ensure the directory for captured images exists
        create_directory(dir_name)

        frame_counter += 1
        ret, frame = camera.read()
        ret, saving_frame = camera.read()

        height, width, _ = saving_frame.shape

        # Draw height and width on the saving frame
        draw_text(saving_frame, f"Height: {height}", (30, 50))
        draw_text(saving_frame, f"Width: {width}", (30, 70))

        if capture_image and cap_frame <= number_image_captured:
            cap_frame += 1
            # Indicate capturing on the frame and save the image
            draw_text(frame, 'Capturing', (50, 70), font_scale=2, color=(0, 244, 255), thickness=1)
            cv2.imwrite(f"{dir_name}/frame-{cap_frame}.png", saving_frame)
            capture_image = False
        else:
            # Indicate not capturing on the frame
            draw_text(frame, 'Not Capturing', (50, 70), font_scale=2, color=(255, 0, 255), thickness=1)
            capture_image = False

        # Display frames
        cv2.imshow("frame", frame)
        cv2.imshow("saving Image", saving_frame)

        total_time = time.time()
        frame_time = total_time - starting_time

        # Calculate frames per second
        fps = frame_counter / frame_time

        # Check for key presses
        if cv2.waitKey(1) == ord('c'):
            capture_image = True

        if cv2.waitKey(1) in [ord('q'), 27]:
            break

    # Release the camera and close windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run the main function if the script is executed directly
    main()

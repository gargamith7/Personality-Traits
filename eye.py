import cv2
from mtcnn.mtcnn import MTCNN

def extract_eye_features(image_rgb):
    # Create an MTCNN detector instance
    detector = MTCNN()
    # Detect faces in the image
    faces = detector.detect_faces(image_rgb)
    try :
        # Assuming there's only one face in the image, or you can choose the largest face
        face = faces[0]
        # Get the facial landmark points
        landmarks = face['keypoints']
        # Extract the coordinates of the left and right eyes
        left_eye_x, left_eye_y = landmarks['left_eye']
        right_eye_x, right_eye_y = landmarks['right_eye']
        # Define the width and height of the eye region
        eye_width = 20
        eye_height = 20
        # Calculate the top-left and bottom-right coordinates for cropping the left eye
        left_eye_top_left_x = left_eye_x - eye_width // 2
        left_eye_top_left_y = left_eye_y - eye_height // 2
        left_eye_bottom_right_x = left_eye_x + eye_width // 2
        left_eye_bottom_right_y = left_eye_y + eye_height // 2
        # Crop the left eye region from the image
        left_eye_img = image_rgb[left_eye_top_left_y:left_eye_bottom_right_y, left_eye_top_left_x:left_eye_bottom_right_x]
        # Calculate the top-left and bottom-right coordinates for cropping the right eye
        right_eye_top_left_x = right_eye_x - eye_width // 2
        right_eye_top_left_y = right_eye_y - eye_height // 2
        right_eye_bottom_right_x = right_eye_x + eye_width // 2
        right_eye_bottom_right_y = right_eye_y + eye_height // 2
        # Crop the right eye region from the image
        right_eye_img = image_rgb[right_eye_top_left_y:right_eye_bottom_right_y, right_eye_top_left_x:right_eye_bottom_right_x]
        return left_eye_img, right_eye_img
    except:
        return image_rgb, image_rgb
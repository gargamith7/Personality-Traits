import cv2
from mtcnn.mtcnn import MTCNN

def extract_mouth_features(image_rgb):
    # Create an MTCNN detector instance
    detector = MTCNN()
    # Detect faces in the image
    faces = detector.detect_faces(image_rgb)
    try:
        # Assuming there's only one face in the image, or you can choose the largest face
        face = faces[0]
        # Get the facial landmark points
        landmarks = face['keypoints']
        # Extract the coordinates of the mouth
        mouth_x, mouth_y = landmarks['mouth_left'], landmarks['mouth_right']
        # Define the width and height of the mouth region
        mouth_width = 20
        mouth_height = 20
        # Calculate the top-left and bottom-right coordinates for cropping the mouth
        mouth_top_left_x = mouth_x[0] - mouth_width // 2
        mouth_top_left_y = mouth_y[1] - mouth_height // 2
        mouth_bottom_right_x = mouth_x[1] + mouth_width // 2
        mouth_bottom_right_y = mouth_y[1] + mouth_height // 2
        # Crop the mouth region from the image
        mouth_img = image_rgb[mouth_top_left_y:mouth_bottom_right_y, mouth_top_left_x:mouth_bottom_right_x]
        return mouth_img
    except:
        return image_rgb
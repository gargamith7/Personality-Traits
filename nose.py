import cv2
from mtcnn.mtcnn import MTCNN

def extract_nose_feature(image_rgb):
    # Create an MTCNN detector instance
    detector = MTCNN()
    # Detect faces in the image
    faces = detector.detect_faces(image_rgb)
    # Assuming there's only one face in the image, or you can choose the largest face
    try :
        face = faces[0]
        # Get the facial landmark points
        landmarks = face['keypoints']
        # Extract the coordinates of the nose
        nose_x, nose_y = landmarks['nose']
        # Define the width and height of the nose region
        nose_width = 20
        nose_height = 20
        # Calculate the top-left and bottom-right coordinates for cropping the nose
        nose_top_left_x = nose_x - nose_width // 2
        nose_top_left_y = nose_y - nose_height // 2
        nose_bottom_right_x = nose_x + nose_width // 2
        nose_bottom_right_y = nose_y + nose_height // 2
        # Crop the nose region from the image
        nose_img = image_rgb[nose_top_left_y:nose_bottom_right_y, nose_top_left_x:nose_bottom_right_x]
        return nose_img
    except:
        return image_rgb

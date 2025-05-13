import os
import cv2
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from eye import extract_eye_features
from mouth import extract_mouth_features
from nose import extract_nose_feature


# Reading Image
def Read_Image(Filename):
    Image = cv2.imread(Filename)
    Image = cv2.resize(Image, (256, 256))
    Image = np.uint8(Image)
    return Image


# Read Dataset1
an = 0
if an == 1:
    Images = []
    Target = []
    dir = "./Datasets/Dataset1/"
    listdir = os.listdir(dir)
    for i in range(len(listdir)):
        subfolder = dir + listdir[i]
        listdir2 = os.listdir(subfolder)
        for j in range(len(listdir2)):
            if j < 200:
                Filename = subfolder + "/" + listdir2[j]
                img = Read_Image(Filename)
                Images.append(img)
                Target.append(i)

    label_encoder = LabelEncoder()
    Target = label_encoder.fit_transform(Target)

    Target = to_categorical(Target)

    np.save("Images_1.npy", np.array(Images))
    np.save("Target_1.npy", Target)
# Read dataset2


an = 0
if an == 1:
    Images = []
    Target = []
    dir = "./Datasets/Dataset2/"
    listdir = os.listdir(dir)
    for i in range(len(listdir)):
        subfolder = dir + listdir[i]
        listdir2 = os.listdir(subfolder)
        for j in range(len(listdir2)):
            if j < 200:
                Filename = subfolder + "/" + listdir2[j]
                img = Read_Image(Filename)
                Images.append(img)
                Target.append(i)

    label_encoder = LabelEncoder()
    Target = label_encoder.fit_transform(Target)

    Target = to_categorical(Target)

    np.save("Images_2.npy", np.array(Images))
    np.save("Target_2.npy", Target)

# Viola Jones-based face detection

an = 0
if an == 1:
    for i in range(2):
        # Load preprocessed data
        Data = np.load(f'Images_{i + 1}.npy', allow_pickle=True)
        Target = np.load(f'Target_{i + 1}.npy', allow_pickle=True)

        # Load the pre-trained Viola-Jones face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        detected_faces = []
        for img in Data:
            # Convert to grayscale for the cascade classifier
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Perform face detection
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(10, 10))

            # Optionally, draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            detected_faces.append(img)
        # Save results if needed
        np.save(f'Detected_Faces_{i + 1}.npy', detected_faces)
        print(f"Faces detected and saved for Images_{i + 1}.npy")

an = 1
if an == 1:
    for n in range(2):  # Loop runs only once
        left_eye_imgs = []
        right_eye_imgs = []
        nose_imgs = []
        mouth_imgs = []

        Image = np.load(f'Detected_Faces_{n + 1}.npy', allow_pickle=True)
        for i, image in enumerate(Image):

            left_eye_img, right_eye_img = extract_eye_features(image)
            # nose_img = extract_nose_feature(image)
            # mouth_img = extract_mouth_features(image)

            left_eye_img = cv2.resize(left_eye_img, (20, 20))

            right_eye_img = cv2.resize(right_eye_img, (20, 20))

            # nose_img = cv2.resize(nose_img, (20, 20))

            # mouth_img = cv2.resize(mouth_img, (20, 20))

            left_eye_imgs.append(left_eye_img)
            right_eye_imgs.append(right_eye_img)
            # nose_imgs.append(nose_img)
            # mouth_imgs.append(mouth_img)

        np.save(f'Left_Eye_{n + 1}.npy', left_eye_imgs)
        np.save(f'Right_Eye_{n + 1}.npy', right_eye_imgs)
        # np.save(f'Nose_{n + 1}.npy', nose_imgs)
        # np.save(f'Mouth_{n + 1}.npy', mouth_imgs)

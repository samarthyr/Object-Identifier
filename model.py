from sklearn.svm import LinearSVC
import numpy as np
import cv2 as cv
from PIL import Image

class Model:

    def __init__(self):
        self.model = LinearSVC()

    def train_model(self, counters):
        img_list = []
        class_list = []

        for i in range(1, counters[0]):
            img = cv.imread(f'1/frame{i}.jpg')[:, :, 0]
            img = cv.resize(img, (150, 150))  # Ensure image is resized to 150x150
            img = img.flatten()  # Flatten the image
            img_list.append(img)
            class_list.append(1)

        for i in range(1, counters[1]):
            img = cv.imread(f'2/frame{i}.jpg')[:, :, 0]
            img = cv.resize(img, (150, 150))  # Ensure image is resized to 150x150
            img = img.flatten()  # Flatten the image
            img_list.append(img)
            class_list.append(2)

        img_list = np.array(img_list)
        class_list = np.array(class_list)
        self.model.fit(img_list, class_list)
        print("Model successfully trained!")

    def predict(self, frame):
        ret, frame = frame
        if ret:
            # Convert frame to grayscale
            gray_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            # Save the frame as an image
            cv.imwrite("frame.jpg", gray_frame)
            # Open and resize the image using PIL
            img = Image.open("frame.jpg")
            img.thumbnail((150, 150), Image.Resampling.LANCZOS)
            img.save("frame.jpg")
            # Read the image back in grayscale and flatten it
            img = cv.imread('frame.jpg', cv.IMREAD_GRAYSCALE)
            img = cv.resize(img, (150, 150))  # Ensure image is resized to 150x150
            img = img.flatten()  # Flatten the image
            # Predict the class
            prediction = self.model.predict([img])
            return prediction[0]
        return None

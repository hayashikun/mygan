from os import path

import cv2
import requests
from mygan import TmpFilePath


class FaceRecognizer:
    def __init__(self):
        cascade_file = "haarcascade_frontalface_alt.xml"
        cascade_file_path = path.join(TmpFilePath, cascade_file)
        if not path.exists(cascade_file_path):
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{}".format(cascade_file)
            response = requests.get(url)
            with open(cascade_file_path, "wb") as f:
                f.write(response.content)
        self.cascade = cv2.CascadeClassifier(cascade_file_path)

    def recognize(self, image, min_size=None, max_size=None):
        if isinstance(image, str) and path.exists(image):
            image = cv2.imread(image)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(image_gray, scaleFactor=1.11, minNeighbors=3,
                                              minSize=min_size, maxSize=max_size)
        return faces

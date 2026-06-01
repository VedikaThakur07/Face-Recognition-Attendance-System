import cv2
import numpy as np
from PIL import Image
import os

path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces = []
ids = []

for file in os.listdir(path):
    img = Image.open(os.path.join(path, file)).convert('L')
    img_np = np.array(img, 'uint8')

    id = int(file.split('.')[1])

    detected = detector.detectMultiScale(img_np)
    for (x,y,w,h) in detected:
        faces.append(img_np[y:y+h,x:x+w])
        ids.append(id)

recognizer.train(faces, np.array(ids))
recognizer.save('trainer/trainer.yml')

print("Training Done")
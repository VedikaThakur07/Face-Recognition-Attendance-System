import tkinter as tk
from tkinter import messagebox
import cv2
import os

def capture_images():
    face_id = entry_id.get()
    name = entry_name.get()

    if face_id == "" or name == "":
        messagebox.showerror("Error", "Enter ID and Name")
        return

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    count = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            count += 1
            cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y+h,x:x+w])
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow('Capturing Faces', img)

        if cv2.waitKey(1) == 27 or count >= 40:
            break

    cam.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Done", "Images Captured")

root = tk.Tk()
root.title("Face Attendance System")

tk.Label(root, text="Enter ID").pack()
entry_id = tk.Entry(root)
entry_id.pack()

tk.Label(root, text="Enter Name").pack()
entry_name = tk.Entry(root)
entry_name.pack()

tk.Button(root, text="Capture Images", command=capture_images).pack()

root.mainloop()
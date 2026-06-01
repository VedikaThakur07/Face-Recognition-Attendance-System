import tkinter as tk
import cv2
import csv
from datetime import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

names = ['None', 'Vedika']   # ID=1 → Vedika

def mark_attendance():
    cam = cv2.VideoCapture(0)

    marked = set()

    while True:
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,1.3,5)

        for(x,y,w,h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            if confidence < 70:
                name = names[id]
            else:
                name = "Unknown"

            # 📅 date + time
            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            # CSV save (only once)
            if name != "Unknown" and name not in marked:
                with open('attendance.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([name, date, time])
                marked.add(name)

            # 📦 box + text
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img, f"{name}", (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow('Attendance System', img)

        # ❌ video बंद करने का button
        if cv2.waitKey(1) == 27:   # ESC दबाओ
            break

    cam.release()
    cv2.destroyAllWindows()

# GUI
root = tk.Tk()
root.title("Attendance System")

tk.Button(root, text="Start Attendance", command=mark_attendance).pack()

root.mainloop()
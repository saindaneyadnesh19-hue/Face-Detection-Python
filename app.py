import tkinter as tk
from tkinter import messagebox
import cv2


def start_recognition():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            messagebox.showerror("Error", "Camera could not be accessed")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]

            face_id, confidence = recognizer.predict(face)

            if confidence < 70:
                name = "Yadnyesh"
            else:
                name = "Unknown"

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                name,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        cv2.imshow("Live Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def exit_app():
    root.destroy()


# Main GUI window
root = tk.Tk()
root.title("Face Recognition System")
root.geometry("500x350")

title = tk.Label(
    root,
    text="Face Recognition System",
    font=("Arial", 24, "bold")
)
title.pack(pady=40)

subtitle = tk.Label(
    root,
    text="Python + OpenCV",
    font=("Arial", 14)
)
subtitle.pack(pady=10)

start_button = tk.Button(
    root,
    text="▶ Start Face Recognition",
    font=("Arial", 14),
    command=start_recognition,
    width=25,
    height=2
)
start_button.pack(pady=20)

exit_button = tk.Button(
    root,
    text="❌ Exit",
    font=("Arial", 12),
    command=exit_app,
    width=15
)
exit_button.pack(pady=10)

root.mainloop()
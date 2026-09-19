import cv2
import os

name = "Yadnyesh"

# Create dataset folder
folder = "dataset"
os.makedirs(folder, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)

count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera could not be accessed")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:
        count += 1

        face = gray[y:y+h, x:x+w]

        filename = f"{folder}/{name}_{count}.jpg"
        cv2.imwrite(filename, face)

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Photo: {count}/30",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Dataset", frame)

    # Stop after 30 photos
    if cv2.waitKey(100) & 0xFF == ord("q") or count >= 30:
        break

cap.release()
cv2.destroyAllWindows()

print("Dataset created successfully!")
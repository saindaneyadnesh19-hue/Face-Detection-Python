import cv2
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()

data_path = "dataset"
faces = []
ids = []

for file in os.listdir(data_path):
    if file.endswith(".jpg"):
        path = os.path.join(data_path, file)

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        faces.append(image)
        ids.append(1)

recognizer.train(faces, __import__("numpy").array(ids))

recognizer.save("trainer.yml")

print("Face training completed successfully!")
print("Model saved as trainer.yml")
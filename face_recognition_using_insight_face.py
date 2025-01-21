import insightface
import cv2
import numpy as np
from insightface.app import FaceAnalysis


"""
Face Recognition and Registration using InsightFace

This script registers known faces from image files, performs real-time face recognition using the webcam, 
and displays the recognized faces along with their bounding boxes and landmarks.

Modules:
--------
- insightface: The InsightFace library for face detection and recognition.
- cv2: OpenCV library for image processing and video capture.
- numpy: Library for handling arrays and performing numerical operations.

Functions:
----------
- register_face(name, image_path):
    Registers a new face by detecting it in the provided image and storing its embedding for future recognition.

Process:
--------
1. Initializes the InsightFace face detection and recognition model.
2. Registers known faces (name and image).
3. Captures real-time video using OpenCV from the webcam.
4. Detects faces in each frame and compares them against the registered faces using cosine similarity.
5. Draws landmarks (eyes, nose) and bounding box around recognized faces.
6. Displays recognized faces with their name, recognition confidence, and bounding box on the video feed.
7. Allows quitting the program by pressing the "q" key.

Details:
--------
- Face landmarks (eyes, nose) are drawn on the recognized faces for visualization.
- Recognition confidence score (cosine similarity) is displayed next to the recognized faces.
- Bounding box dimensions (height and width) are calculated and used for drawing a colored label showing the recognized name and confidence.
- Face embeddings for registered faces are stored in a dictionary and used for recognition.

Usage:
------
1. The script continuously captures webcam frames, detects faces, and compares them to registered faces.
2. Press "q" to exit the webcam feed.

Dependencies:
------------
- insightface
- opencv-python
- numpy

"""


## registering a face
known_faces = {}

def register_face(name, image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)
    if faces:
        known_faces[name] = faces[0].normed_embedding
        print(f"Registered {name}")
    else:
        print(f"No face detected in {image_path}")
    

cap = cv2.VideoCapture(0)
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=1)

register_face("Arunisto", "arunisto.jpeg")
register_face("Elon Musk", "elon_musk.jpg")
register_face("Steve Jobs", "jobs.jpeg")

while True:
    _, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb_frame)
    if faces:
        for face in faces:
            # print(faces)
            #--------------------- Recognising Faces -----------------#
            best_match = None
            best_score = float("-inf")
            for name, embedding in known_faces.items():
                score = np.dot(face.normed_embedding, embedding) #cosine similarity
                if score > best_score:
                    best_score = score
                    best_match = name
            #--------------------- drawing landmarks -----------------#
            left_eye, right_eye = face.kps[0].astype(int), face.kps[1].astype(int)
            lex, ley = left_eye
            rex, rey = right_eye
            cv2.circle(frame, (lex, ley), 3, (0, 255, 255), 3)
            cv2.circle(frame, (rex, rey), 3, (0, 255, 255), 3)
            nose = face.kps[2].astype(int)
            nx, ny = nose
            cv2.circle(frame, (nx, ny), 3, (0, 255, 0), 3)
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #height and width of bbox
            if best_match and best_score > 0.3:
                w = x2-x1
                h = y2-y1
                cv2.rectangle(frame, (x1, y1+h), (x1+w, y1+h+30), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, f"{best_match}", (x1+5, y1+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"{best_score:.2f}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            #--------------------------------------------------------#
    cv2.imshow("window", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
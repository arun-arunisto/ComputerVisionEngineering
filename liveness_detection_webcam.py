import cv2
import numpy as np
import face_recognition
import tensorflow as tf
from tensorflow.keras.models import load_model



liveness_model = load_model("/home/royalbrothers/project/face_match/liveness_model.keras")
cap = cv2.VideoCapture(0)

process_this_frame = True

while True:
    _, frame = cap.read()

    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)

    process_this_frame = not process_this_frame
    if face_locations:
        for (top, right, bottom, left) in face_locations:
            top*=4
            right*=4
            bottom*=4
            left*=4
            # face_image = frame[top:bottom, left:right]
            face_image_resize = cv2.resize(frame, (224, 224))
            face_image_resize = face_image_resize/255.0
            face_image_resize = np.expand_dims(face_image_resize, axis=0)
            liveness_score = liveness_model.predict(face_image_resize)[0][0]
            label = "Real"
            if liveness_score < 0.9:
                label = "Spoof"
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, label, (left+6, bottom-6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
    cv2.imshow("Window", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

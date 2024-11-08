import cv2
import face_recognition
import numpy as np

cap = cv2.VideoCapture(0)

def is_looking_at_camera(face_landmarks):
    #determining the person is looking at the camera or not
    #eye landmarks
    left_eye = face_landmarks["left_eye"]
    right_eye = face_landmarks["right_eye"]

    #nose landmarks
    nose_bridge = face_landmarks["nose_bridge"]
    
    #calculating the center of the eye
    left_eye_center = np.mean(left_eye, axis=0).astype(int)
    right_eye_center = np.mean(right_eye, axis=0).astype(int)

    #calculating the nose bridge
    nose_center = np.mean(nose_bridge, axis=0).astype(int)

    #visualizing the centers for debugging
    cv2.circle(frame, tuple(left_eye_center), 2, (255, 0, 0), 2)
    cv2.circle(frame, tuple(right_eye_center), 2, (255, 0, 0), 2)
    cv2.circle(frame, tuple(nose_center), 2, (255, 0, 0), 2)
    #calculating the horizontal alignment by comparing eye centers and nose bridge
    print("left eye:",abs(left_eye_center[0] - nose_bridge[0][0]))
    print("right eye:",abs(right_eye_center[0] - nose_bridge[0][0]))
    # right_eye_tol = 40
    # left_eye_tol = 55
    # result = {}
    # if left_eye_center[0] > 60 and right_eye_center[0] > 60:
    #     result.update({"Near to the camera":True})
    
    # if left_eye_center[0] < 40 and right_eye_center[0] < 40:
    #     result.update({"Far away from camera":True})
    
    # left_eye_aligned = abs(left_eye_center[0] - nose_bridge[0][0]) < left_eye_tol and abs(left_eye_center[0] - nose_bridge[0][0]) > right_eye_tol
    # right_eye_aligned = abs(right_eye_center[0] - nose_bridge[0][0]) > right_eye_tol and abs(right_eye_center[0] - nose_bridge[0][0]) < left_eye_tol

    
    # if left_eye_aligned and right_eye_aligned:
    #     result.update({"is_looking_at_camera":True})
    
    print((abs(left_eye_center[0] - nose_bridge[0][0]) - abs(right_eye_center[0] - nose_bridge[0][0])))
    
    looking_at_camera = (abs(left_eye_center[0] - nose_bridge[0][0]) - abs(right_eye_center[0] - nose_bridge[0][0])) < 10
    #if the both eye aligns to the nose bridge then the person is looking at the camera
    return looking_at_camera

while True:
    _, frame = cap.read()
    face_locations = face_recognition.face_locations(frame)
    face_landmarks_list = face_recognition.face_landmarks(frame)

    # loop over each face found
    for face_landmarks in face_landmarks_list:
        print(face_landmarks)
        #drawing a box around the face
        top, right, bottom, left = face_locations[face_landmarks_list.index(face_landmarks)]
        cv2. rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        #checking if the person is looking at the camera
        if is_looking_at_camera(face_landmarks):
            cv2.rectangle(frame, (left, bottom-35),(right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, "Looking at camera", (left+6, bottom-6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.rectangle(frame, (left, bottom-35),(right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, "Not looking at camera", (left+6, bottom-6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        # result = is_looking_at_camera(face_landmarks)
        # if result:
        #     try:
        #         if result["is_looking_at_camera"]:
        #             cv2.rectangle(frame, (left, bottom-35),(right, bottom), (0, 0, 255), cv2.FILLED)
        #             cv2.putText(frame, "Looking at camera", (left+6, bottom-6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        #         elif result["Near to the camera"]:
        #             cv2.rectangle(frame, (left, bottom-35),(right, bottom), (0, 0, 255), cv2.FILLED)
        #             cv2.putText(frame, "Near to the camera", (left+6, bottom-6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        #         elif result["Far away from camera"]:
        #             cv2.rectangle(frame, (left, bottom-35),(right, bottom), (0, 0, 255), cv2.FILLED)
        #             cv2.putText(frame, "Far away from camera", (left+6, bottom-6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        #         else:
        #             cv2.rectangle(frame, (left, bottom-35),(right, bottom), (0, 0, 255), cv2.FILLED)
        #             cv2.putText(frame, "Not looking at camera", (left+6, bottom-6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        #     except Exception as e:
        #         continue
    cv2.imshow("Window", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

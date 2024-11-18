import cv2
from scipy.spatial import distance as dist
import numpy as np
import dlib
from imutils import face_utils
import imutils
from collections import deque



#EAR - eye aspect ratio
def eye_aspect_ratio(eye):
    #computing the euclidean distances between the two sets of vertical eye landmarks (x, y) - coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    #computing the eye aspect ratio
    ear = (A + B) / (2.0*C)
    #returning the eye aspect ratio
    return ear

cap = cv2.VideoCapture(0)

#setting the ear value
ear_values = deque(maxlen=5)

#defining two constants, one for the eye aspect ratio to indicate blink
EYE_AR_THRESH = 0.3
#the second constant for the number of consecutive frames the eye must be below the threshold
EYE_AR_CONSEC_FRAMES = 3
#initializing the frame counters and total number of blinks
COUNTER = 0
TOTAL = 0

#facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/royalbrothers/tutorial/computer_vision_engineering/shape_predictor_68_face_landmarks.dat")

#right eye coordinates
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
#left eye coordinates
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]

while True:
    _, frame = cap.read()
    #converting the frame to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detecting face in grayscale frame
    rects = detector(gray, 0)
    if rects:
        for rect in rects:
            #  cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
            # using the facce cordinates for shape prediction
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # print(shape)
            # extracting the left and right eye coordinates
            leftEye = shape[lstart:lend]
            rightEye = shape[rstart:rend]
            # print(leftEye, rightEye)
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            #avaerage the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # smoothing EAR values
            ear_values.append(ear)
            smoothed_ear = sum(ear_values) / len(ear_values)

            #computing the convex hull for the left and right eye
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            #visualizing the each of the eye
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            
            #setting the blink counter
            if smoothed_ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                #resetting the eye frame counter
                COUNTER = 0
            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(smoothed_ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.imshow("Window", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
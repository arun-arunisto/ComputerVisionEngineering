import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

"""
Face Detection and Analysis using InsightFace and OpenCV

This script captures video from a webcam, detects faces using the InsightFace library, 
and overlays various facial features such as bounding boxes, key points, landmarks, 
detection scores, pose estimation, and more.

Features:
----------
1. Face Detection:
   - Detects faces in real-time using InsightFace.
   - Draws bounding boxes around detected faces.

2. Key Facial Features (5-point landmarks):
   - Detects and marks left eye, right eye, nose, mouth corners.

3. 3D and 2D Facial Landmarks:
   - Marks 68 3D facial landmarks (white dots).
   - Marks 106 2D facial landmarks (red dots).

4. Pose Estimation:
   - Estimates and displays yaw (left/right), pitch (up/down), and roll (tilt).

5. Detection Confidence Score:
   - Displays detection confidence score as a percentage.

6. Real-Time Display:
   - Continuously processes frames from the webcam and displays results.
   - Press 'q' to exit the program.

Usage:
------
- Ensure OpenCV, NumPy, and InsightFace are installed.
- Run the script and allow webcam access.
- The analyzed video stream will be displayed with facial annotations.

Dependencies:
-------------
- OpenCV (`cv2`)
- NumPy (`numpy`)
- InsightFace (`insightface`)
"""



#initializing camera
cap = cv2.VideoCapture(0)
#initializing face analysis
app = FaceAnalysis(providers=['CPUExecutionProvider']) # using cpu for inference if GPU available use 'GPUExecutionProvider'

app.prepare(ctx_id=1, det_size=(640, 640)) #using ctx_id=1 for cpu

while True:
    _, frame = cap.read()
    #converting bgr to rgb
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #detecting faces
    faces = app.get(rgb_frame)
    if faces:
        #drawing bbox
        for face in faces:
            print(face)
            #------------- eyes ---------------------------#
            left_eye, right_eye = face.kps[0].astype(int), face.kps[1].astype(int)
            lex, ley = left_eye
            rex, rey = right_eye
            cv2.circle(frame, (lex, ley), 3, (0, 0, 255), 3)
            cv2.circle(frame, (rex, rey), 3, (0, 0, 255), 3)
            #----------------------------------------------#
            #---------- nose ------------------------------#
            nose = face.kps[2].astype(int)
            nx, ny = nose
            cv2.circle(frame, (nx, ny), 3, (255, 0, 0), 3)
            #----------------------------------------------#
            #----------- mouth ----------------------------#
            mouth_left, mouth_right = face.kps[3].astype(int), face.kps[4].astype(int)
            mlx, mly = mouth_left
            mrx, mry = mouth_right
            cv2.circle(frame, (mlx, mly), 3, (0, 255, 255), 3)
            cv2.circle(frame, (mrx, mry), 3, (0, 255, 255), 3)
            #-----------------------------------------------#
            #------------ bounding box ---------------------#
            box = face.bbox.astype(int) #converting bbox to integer
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #------------------ detection score -------------#
            cv2.putText(frame, f"Detection Score: {face.det_score:.2f} %", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #-----------------------------------------------#
            #------------------ 3D facial landmarks --------#
            for (x, y, z) in face.landmark_3d_68.astype(int):  # Convert to int
                cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)  # white dots
            #-----------------------------------------------#
            #------------------ pose -----------------------#
            """
            Index	  Pose Type	            Description
            pose[0]	  Yaw (Left/Right)	    Negative = Left, Positive = Right
            pose[1]	  Pitch (Up/Down)	    Negative = Looking Down, Positive = Looking Up
            pose[2]	  Roll (Tilt)	        Negative = Left Tilt, Positive = Right Tilt
            """
            yaw, pitch, roll = face.pose
            # Display pose values on the frame
            cv2.putText(frame, f"Yaw: {yaw:.2f}", (x1, y2 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 2)
            cv2.putText(frame, f"Pitch: {pitch:.2f}", (x1, y2 + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 2)
            cv2.putText(frame, f"Roll: {roll:.2f}", (x1, y2+60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 2)
            #----------------------------------------------#
            #------------------ 2d 106 face landmarks -----#
            for (x, y) in face.landmark_2d_106.astype(int):  # Convert to int
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)  # dots
            #----------------------------------------------#
    
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

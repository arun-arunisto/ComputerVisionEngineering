#detecting faces using yolo
import cv2
import numpy as np
from ultralytics import YOLO
from enum import Enum
from deepface.models.Detector import Detector, FacialAreaRegion
import gdown
import os
class YoloModel(Enum):
    V8N = 0
    V11N = 1
    V11S = 2
    V11M = 3

# Model's weights paths
WEIGHT_NAMES = ["yolov8n-face.pt",
                "yolov11n-face.pt",
                "yolov11s-face.pt",
                "yolov11m-face.pt"]

def download_weights_if_necessary(source_url, file_name):
    if os.path.exists(f'/home/royalbrothers/tutorial/computer_vision_engineering/deepFaceModule/{file_name}'):
        return file_name
    try:
        gdown.download(source_url, file_name, quiet=False)
        return file_name
    except Exception as e:
        return e

# Google Drive URL from repo (https://github.com/derronqi/yolov8-face) ~6MB
WEIGHT_URLS = ["https://drive.google.com/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb",
            "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11n-face.pt",
            "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11s-face.pt",
            "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11m-face.pt"]

class YoloFaceDetector:
    def __init__(self, model):
        super().__init__()
        self.model = self.build_model(model)
    
    def build_model(self, model):
        weight_file = download_weights_if_necessary(
            file_name=WEIGHT_NAMES[model.value], source_url=WEIGHT_URLS[model.value]
        )
        return YOLO(weight_file)

    def detect_faces(self, img: np.ndarray):
        """
        Detect and align face with yolo

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        resp = []

        # Detect faces
        results = self.model.predict(
            img,
            verbose=False,
            show=False,
            conf=float(0.5),
        )[0]

        # For each face, extract the bounding box, the landmarks and confidence
        for result in results:

            if result.boxes is None:
                continue

            # Extract the bounding box and the confidence
            x, y, w, h = result.boxes.xywh.tolist()[0]
            confidence = result.boxes.conf.tolist()[0]

            right_eye = None
            left_eye = None

            # yolo-facev8 is detecting eyes through keypoints,
            # while for v11 keypoints are always None
            if result.keypoints is not None:
                # right_eye_conf = result.keypoints.conf[0][0]
                # left_eye_conf = result.keypoints.conf[0][1]
                right_eye = result.keypoints.xy[0][0].tolist()
                left_eye = result.keypoints.xy[0][1].tolist()

                # eyes are list of float, need to cast them tuple of int
                left_eye = tuple(int(i) for i in left_eye)
                right_eye = tuple(int(i) for i in right_eye)

            x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
            facial_area = FacialAreaRegion(
                x=x,
                y=y,
                w=w,
                h=h,
                left_eye=left_eye,
                right_eye=right_eye,
                confidence=confidence,
            )
            resp.append(facial_area)

        return resp

class YoloDetectorClientV8n(YoloFaceDetector):
    def __init__(self):
        super().__init__(YoloModel.V8N)


class YoloDetectorClientV11n(YoloFaceDetector):
    def __init__(self):
        super().__init__(YoloModel.V11N)


class YoloDetectorClientV11s(YoloFaceDetector):
    def __init__(self):
        super().__init__(YoloModel.V11S)


class YoloDetectorClientV11m(YoloFaceDetector):
    def __init__(self):
        super().__init__(YoloModel.V11M)

def main():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        frame_v11n = frame.copy()
        frame_v11s = frame.copy()
        frame_v11m = frame.copy()
        #checking with yolov8
        detector_v8 = YoloDetectorClientV8n()
        faces = detector_v8.detect_faces(frame)
        if faces:
            for i in faces:
                x, y, w, h = i.x, i.y, i.w, i.h
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.rectangle(frame, (x, y+h), (x+w, y+h+30), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, f"YoloV8 - {i.confidence:.2f}%", (x+5, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        #checking with yolov11n
        detector_v11n = YoloDetectorClientV11n()
        faces_v11n = detector_v11n.detect_faces(frame_v11n)
        if faces_v11n:
            for i in faces_v11n:
                x, y, w, h = i.x, i.y, i.w, i.h
                cv2.rectangle(frame_v11n, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.rectangle(frame_v11n, (x, y+h), (x+w, y+h+30), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame_v11n, f"YoloV11n - {i.confidence:.2f}%", (x+5, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        #checking with yolov11s
        detector_v11s = YoloDetectorClientV11s()
        faces_v11s = detector_v11s.detect_faces(frame_v11s)
        if faces_v11s:
            for i in faces_v11s:
                x, y, w, h = i.x, i.y, i.w, i.h
                cv2.rectangle(frame_v11s, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.rectangle(frame_v11s, (x, y+h), (x+w, y+h+30), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame_v11s, f"YoloV11s - {i.confidence:.2f}%", (x+5, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        #checking with yolov11s
        detector_v11m = YoloDetectorClientV11m()
        faces_v11m = detector_v11m.detect_faces(frame_v11m)
        if faces_v11m:
            for i in faces_v11m:
                x, y, w, h = i.x, i.y, i.w, i.h
                cv2.rectangle(frame_v11m, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.rectangle(frame_v11m, (x, y+h), (x+w, y+h+30), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame_v11m, f"YoloV11m - {i.confidence:.2f}%", (x+5, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("yolov8", frame)
        cv2.imshow("yolov11n", frame_v11n)
        cv2.imshow("yolov11s", frame_v11s)
        cv2.imshow("yolov11m", frame_v11m)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

if __name__ == "__main__":
    main()
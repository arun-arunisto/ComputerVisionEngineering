from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("path/to/train/weights/last.pt")

img = cv2.imread("path/to/your/image")

result = model(img, show=True)

key = cv2.waitKey(0)
if key == ord("q"):
    cv2.destroyAllWindows()

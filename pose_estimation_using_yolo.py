import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture("/home/royalbrothers/yolo_projects/dancers.mp4")
model = YOLO("yolo11n-pose.pt")

skeleton = [
    (1, 3),  # Left eye to left ear
    (2, 4),  # Right eye to right ear
    (1, 0), (0, 2),  # Eyes to nose
    (3, 4),   # Left ear to right ear (approximate head line)
    (5, 7), (7, 9), # left arm
    (6, 8), (8, 10), # right arm
    (5, 6), # shoulders
    (11, 12), # hips
    (5, 11), (6, 12), # torso
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16), # right leg
]


while True:
    _, frame = cap.read()

    # predicting the model
    results = model(frame)
    for result in results:
        if result.keypoints is not None:
            xy = result.keypoints.xy
            xyn = result.keypoints.xyn  # normalized
            kpts = result.keypoints.data  # x, y, visibility (if available)

            # Draw keypoints
            for kp in xy:
                for x, y in kp:
                    if x > 0 and y > 0:
                        cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), 1)

            # marking lines
            for pt1, pt2 in skeleton:
                if pt1 < len(kp) and pt2 < len(kp):
                    x1, y1 = kp[pt1]
                    x2, y2 = kp[pt2]
                    if all(v > 0 for v in [x1, y1, x2, y2]):
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # blue lines
    # Resize frame for better visibility
    target_width = 1280
    scale = target_width / frame.shape[1]
    frame = cv2.resize(frame, (target_width, int(frame.shape[0] * scale)))
    # Display the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
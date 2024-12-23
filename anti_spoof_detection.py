from deepface import DeepFace
import cv2

def grab_facial_areas(img, detector_backend="opencv", threshold=130, anti_spoofing=False):
    try:
        face_objs = DeepFace.extract_faces(img_path=img, detector_backend=detector_backend, expand_percentage=0, anti_spoofing=anti_spoofing)
        faces = [
            (
                face_obj["facial_area"]["x"],
                face_obj["facial_area"]["y"],
                face_obj["facial_area"]["w"],
                face_obj["facial_area"]["h"],
                face_obj["is_real"],
                face_obj["antispoof_score"]
            )
            for face_obj in face_objs
            if face_obj["facial_area"]["w"] > threshold
        ]
        return faces
    except Exception as e:
        print(str(e))
        return []

def main():
    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        faces = grab_facial_areas(img, anti_spoofing=True)
        if faces:
            print(faces)
            for x, y, w, h, is_real, antispoof_score in faces:
                if is_real:
                    color = (0, 255, 0)
                    real = "Real"
                else:
                    color = (0, 0, 255)
                    real = "Spoof"
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(img, (x, y+h), (x+w, y+h+30), color, cv2.FILLED)
                cv2.putText(img, f"{real} - {antispoof_score:.2f}%", (x+5, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow("frame", img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

if __name__ == "__main__":
    main()
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
            )
            for face_obj in face_objs
            if face_obj["facial_area"]["w"] > threshold
        ]
        return faces
    except Exception as e:
        print(str(e))
        return []


def extracting_facial_areas(img, faces_coordinates):
    detected_faces = []
    for x, y, w, h in faces_coordinates:
        detected_face = img[int(y):int(y+h), int(x):int(x+w)]
        detected_faces.append(detected_face)
    return detected_faces

def perform_demography_analysis(detected_faces):
    for idx, _ in enumerate(detected_faces):
        detected_face = detected_faces[idx]
        demographies = DeepFace.analyze(img_path=detected_face, 
                                        actions=("emotion"),
                                        detector_backend="skip",
                                        enforce_detection=False,
                                        silent=True,)
        if len(demographies) == 0:
            continue
        demography = demographies[0]
        return {"emotion":demography}

def main():
    cap = cv2.VideoCapture(0)
    optimizing_factor = True
    while True:
        _, img = cap.read()
        
        #extracing facial coordinates
        faces = grab_facial_areas(img)
        if faces:
            detected_faces = extracting_facial_areas(img, faces)
            demography_analysis = perform_demography_analysis(detected_faces)
            print("Emotion:", demography_analysis["emotion"])
            emotion = demography_analysis.get("emotion").get("dominant_emotion")
            accuracy = int(demography_analysis["emotion"].get("emotion").get(emotion))
            print("Dominant emotion:",emotion)
            print("Accuracy score:", accuracy)
            for x, y, w, h in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(img, (x, y+h), (x+w, y+h+30), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, f"{emotion} - {accuracy}%", (x+5, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        optimizing_factor = not optimizing_factor
        cv2.imshow("frame", img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

if __name__ == "__main__":
    main()
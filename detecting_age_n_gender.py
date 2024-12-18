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
                                        actions=("age", "gender", "emotion"),
                                        detector_backend="skip",
                                        enforce_detection=False,
                                        silent=True,)
        if len(demographies) == 0:
            continue
        demography = demographies[0]
        return {"age":demography["age"], "gender":demography["dominant_gender"][0:1]}

def age_grouping(age):
    if age >= 0 and age <= 4:
        return "0-4"
    elif age >= 5 and age <= 9:
        return "5-9"
    elif age >= 10 and age <= 14:
        return "10-14"
    elif age >= 15 and age <= 19:
        return "15-19"
    elif age >= 20 and age <= 24:
        return "20-24"
    elif age >= 25 and age <= 29:
        return "25-29"
    elif age >= 30 and age <= 34:    
        return "30-34"
    elif age >= 35 and age <= 39:
        return "35-39"
    elif age >= 40 and age <= 44:
        return "40-44"
    elif age >= 45 and age <= 49:
        return "45-49"
    elif age >= 50 and age <= 54:
        return "50-54"
    elif age >= 55 and age <= 59:
        return "55-59"
    elif age >= 60:
        return "60+"

def gender_classification(gender):
    if gender == "M":
        return "Male"
    elif gender == "F":
        return "Female"


def main():
    cap = cv2.VideoCapture(0)
    optimizing_factor = True
    while True:
        _, img = cap.read()

        #extracting facial coordinates
        faces = grab_facial_areas(img)

        if faces:
            detected_faces = extracting_facial_areas(img, faces)
            demography_analysis = perform_demography_analysis(detected_faces)
            print("Age:",age_grouping(int(demography_analysis["age"])), "Gender:", gender_classification(demography_analysis["gender"]))
            for x, y, w, h in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img, f"Age: {age_grouping(int(demography_analysis['age']))}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, f"Gender: {gender_classification(demography_analysis['gender'])}", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        optimizing_factor = not optimizing_factor

        cv2.imshow("frame", img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

if __name__ == "__main__":
    main()

import face_recognition
import cv2
import numpy as np


#video capture
cap = cv2.VideoCapture(0)

#loading picutre and learn how to recognise it
my_photo = face_recognition.load_image_file("/home/royalbrothers/project/face_match/my_photo.jpg")
my_photo_encoding = face_recognition.face_encodings(my_photo)[0]

#loading another picture
manager_photo = face_recognition.load_image_file("/home/royalbrothers/project/face_match/sajahan.jpeg")
manager_photo_encoding = face_recognition.face_encodings(manager_photo)[0]

#loading another picture
elon_musk_photo = face_recognition.load_image_file("/home/royalbrothers/project/face_match/elon_musk.jpeg")
elon_musk_photo_encoding = face_recognition.face_encodings(elon_musk_photo)[0]

#creating array of known face encodings
known_face_encodings = [
    my_photo_encoding,
    manager_photo_encoding,
    elon_musk_photo_encoding
]

known_face_names = [
    "Arun Arunisto",
    "Mohammad Sajahan",
    "Elon Musk"
]

#initializing some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    _, frame = cap.read()
    if process_this_frame:
        #resizing the frame of video to 1/4 size for faster the face recognition
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        #converting image from bgr color to rgb color
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        #finding the face locations in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            #see if the face is a match for the known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            #if match was found in known face encodings, just use the first one
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # or insted use the know face with the smallest distance to the new one
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
            face_names.append(name)
    process_this_frame = not process_this_frame

    #displaying the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # drawing a bounding box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        #drawing a label with a name below the face 
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left+6, bottom-6), font, 1.0, (255, 255, 255), 1)
    cv2.imshow("Window", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break


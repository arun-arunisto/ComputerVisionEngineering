import face_recognition

#picutre of me
picture_of_me = face_recognition.load_image_file("/home/royalbrothers/project/face_match/my_photo.jpg")
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

# print(my_face_encoding)

#picture to check
unknown_picture = face_recognition.load_image_file("/home/royalbrothers/project/face_match/scarlett_0.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

#checking the face matching or not
results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

if results[0] == True:
    print("It's a picture of me")
else:
    print("It's not a picture of me")
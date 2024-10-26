from django.shortcuts import render
from .models import UsersImages
from PIL import Image
import numpy as np
import face_recognition
from django.core.files.base import ContentFile
import io

# Function to check the photos
def checking_photos(image1_path, image2_array):
    # Load the first image from the file path
    image1 = face_recognition.load_image_file(image1_path)
    
    # Convert the second image (NumPy array) to the correct format
    if image2_array.ndim == 2:  # Grayscale
        image2_array = np.stack((image2_array,) * 3, axis=-1)  # Convert to RGB
    elif image2_array.shape[2] == 4:  # RGBA
        image2_array = image2_array[:, :, :3]  # Convert to RGB by removing alpha channel

    # Get face encodings
    image1_encoding = face_recognition.face_encodings(image1)[0]
    image2_encoding = face_recognition.face_encodings(image2_array)[0]

    # Compare faces
    results = face_recognition.compare_faces([image1_encoding], image2_encoding)

    return results[0]

# View to render index
def index(request):
    return render(request, 'index.html')

# Submitting the form and checking the photo
def submit_form(request):
    if request.method == 'POST':
        name = request.POST['name']
        user_id = request.POST['id']
        uploaded_image = request.FILES.get('photoData')

        if UsersImages.objects.filter(name=name, user_id=user_id).exists():
            user = UsersImages.objects.get(name=name, user_id=user_id)

            # Read the uploaded image directly as a PIL image
            uploaded_image_pil = Image.open(uploaded_image).convert("RGB")
            uploaded_image_np = np.array(uploaded_image_pil)  # Convert PIL image to numpy array

            # Path to the user's image
            user_image_path = user.image.path

            # Check the photos
            if checking_photos(user_image_path, uploaded_image_np):
                print("Photo matches")
                return render(request, 'index.html', {"data": user})
            else:
                return render(request, 'index.html', {"error": "Photo does not match"})

    return render(request, 'index.html', {"message": "No data found"})

from turtle import width
import PIL
from PIL import Image
import os

images_files = os.listdir("/home/royalbrothers/project/CNNPROJECTS/liveness_detection_selfie_dataset/train/not-live")
save_path = "/home/royalbrothers/project/CNNPROJECTS/liveness_detection_selfie_dataset/not_live_compressed_images"

i = 0
for image in images_files:
    try:
        with Image.open(f"/home/royalbrothers/project/CNNPROJECTS/liveness_detection_selfie_dataset/train/not-live/{image}") as img:
            #new width and height
            width = 400
            height = 700

            print("The original size of Image is: ", round(len(img.fp.read())/1024,2), "KB")

            #compressing the image
            img = img.resize((width, height), PIL.Image.NEAREST)

            #saving the image
            img.save(f"{save_path}/{i}.jpg")
            

            with Image.open(f"{save_path}/{i}.jpg") as img:
                print("The compressed size of Image is: ", round(len(img.fp.read())/1024,2), "KB")
        
        i+=1
    except Exception as e:
        print(e)
        continue
    

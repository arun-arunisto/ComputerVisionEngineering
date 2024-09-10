import cv2
import numpy as np
import os
from alive_progress import alive_bar
from time import sleep



def crop_image(image, data, image_key):
    try:
        #getting image height and width
        height, width, channels = image.shape
        #getting x, y, w, h from data
        label, x, y, w, h = data
        #converting to actual pixel values
        bbox_x = float(x) * width
        bbox_y = float(y) * height
        bbox_width = float(w) * width
        bbox_height = float(h) * height
        #finding corners
        top_left_x = int(bbox_x - (bbox_width/2))
        top_left_y = int(bbox_y - (bbox_height / 2))
        bottom_right_x = int(bbox_x + (bbox_width / 2))
        bottom_right_y = int(bbox_y + (bbox_height / 2))
        #cropping the image
        image_copped = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        #resizing the image to (224x224)
        image_resized = cv2.resize(image_copped, (224, 224))
        #saving the image to processed folder
        cv2.imwrite(f"../CNN-DATASET/processed-dataset/{image_key}_224x224.png", image_resized)
        print(f"{image_key}_224x224.png saved successfully")
    except Exception as e:
        print(f"{image_key}_224x224.png not saved due to {e}")

#list outing the images from the dataset
images_list = os.listdir("../CNN-DATASET/digital-data/images")
#print(images_list)
#iterating through the images
i = 0
for image in images_list:
    #print(image)
    #print(image[:-4])
    image_file = cv2.imread(f"../CNN-DATASET/digital-data/images/{image}")
    with open(f"../CNN-DATASET/digital-data/labels/{image[:-4]}.txt", "r") as f:
        data = f.read()
    #converting data to list
    data = data.split(" ")
    if len(data) == 5:
        with alive_bar(100) as bar:
            for i in range(100):
                bar()
        crop_image(image_file, data, image[:-4])
total = os.listdir("../CNN-DATASET/processed-dataset")
with alive_bar(len(total)) as bar:
    for i in range(len(total)):
        sleep(0.01)
        bar()
print(f"Task Completed {len(total)} images saved successfully!!")
        

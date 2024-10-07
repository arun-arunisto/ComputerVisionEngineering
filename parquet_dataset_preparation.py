import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import io

def create_parquet_dataset(data_dir, label_name, return_list):
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg") or filename.endswith("png"):
            file_path = os.path.join(data_dir, filename)

            #opening the image
            with Image.open(file_path) as img:
                #converting the image to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format=img.format)
                img_bytes = img_byte_arr.getvalue()
                label = label_name
                return_list.append([filename, img_bytes, label])


#directory containing braintumor images
giloma_data_dir = "/home/royalbrothers/project/CNNPROJECTS/paraquet_dataset_preparation/brain_tumor/glioma"
healthy_data_dir = "/home/royalbrothers/project/CNNPROJECTS/paraquet_dataset_preparation/brain_tumor/healthy"
meningioma_data_dir = "/home/royalbrothers/project/CNNPROJECTS/paraquet_dataset_preparation/brain_tumor/meningioma"
pituitary_data_dir = "/home/royalbrothers/project/CNNPROJECTS/paraquet_dataset_preparation/brain_tumor/pituitary"


data = []

print("Creating parquet dataset....")
print("Step 1: Converting images to bytes started....")
create_parquet_dataset(giloma_data_dir, "giloma", data)
create_parquet_dataset(healthy_data_dir, "healthy", data)
create_parquet_dataset(meningioma_data_dir, "meningioma", data)
create_parquet_dataset(pituitary_data_dir, "pituitary", data)
print("Step 1: Converting images to bytes completed....")


print("Step 2: Adding image bytes to the dataframe....")
#creating a dataframe with columns for filenames, image bytes and labels
df = pd.DataFrame(data, columns=["filename", "image_bytes", "label"])
print("Step 2: Adding image bytes to the dataframe completed....")

print("Final step: Converting dataframe to parquet....")
#converting dataframe to parquet
df.to_parquet("brain_tumor_dataset.parquet", engine="pyarrow", index=False)
print("Final step: Converting dataframe to parquet completed....")

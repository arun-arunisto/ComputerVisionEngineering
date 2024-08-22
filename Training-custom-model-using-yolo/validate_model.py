from ultralytics import YOLO

#loading trained model
model = YOLO("path/to/train/weights/last.pt")

metrics = model.val()  # assumes `model` has been loaded

print("mean Average Precision 50-90:", metrics.box.map)  # mAP50-95
print("mean Average Precision 50:", metrics.box.map50)  # mAP50
print("mean Average Precision 75:", metrics.box.map75)  # mAP75
print("mean Average Precision 50-95 list:",metrics.box.maps)  # list of mAP50-95 for each category

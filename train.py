import torch
from ultralytics import YOLO

data_yaml = "data.yaml"

model = YOLO("yolo11n.pt")
epochs = 100
batch_size = 16
imgsz = 256
device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)

model.train(data=data_yaml, epochs=epochs, batch=64, imgsz=imgsz)


model.export(format="ncnn", imgsz=imgsz, dynamic=True, simplify=True)

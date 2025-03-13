from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')

data_path = "dataset"

model.train(
    data=data_path,  
    epochs=50,       
    imgsz=224,      
    batch=16,        
    workers=4       

model.val(split="test")

model.export(format='onnx') 

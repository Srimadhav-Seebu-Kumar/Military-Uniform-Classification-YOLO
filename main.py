from ultralytics import YOLO

# Load YOLOv8 classification model
model = YOLO('yolov8n-cls.pt')  # Use a pre-trained classification model

# Define dataset path
data_path = "dataset"

# Train the model
model.train(
    data=data_path,  # Path to dataset
    epochs=50,       # Number of training epochs
    imgsz=224,       # Image size
    batch=16,        # Batch size
    workers=4        # Number of workers for data loading
)

# Validate the model
model.val(split="test")

# Export the model in .pth format
model.export(format='onnx')  # Save trained model

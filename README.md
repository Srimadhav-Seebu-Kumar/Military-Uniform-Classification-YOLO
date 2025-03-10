# YOLOv8 Classification Model Training

## Overview
This project trains a YOLOv8 classification model using a custom dataset. The model is fine-tuned on the provided dataset and validated before being exported in ONNX format.

## Requirements
- Python 3.7+
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- PyTorch
- OpenCV
- NumPy

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/yolov8-classification.git
   cd yolov8-classification
   ```
2. Install dependencies:
   ```bash
   pip install ultralytics torch opencv-python numpy
   ```
3. Ensure your dataset is structured properly in the `dataset/` directory.

## Training the Model
Run the following command to train the YOLOv8 classification model:
```python
from ultralytics import YOLO

# Load YOLOv8 classification model
model = YOLO('yolov8n-cls.pt')  # Pre-trained classification model

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
```

## Validation
To validate the trained model on the test dataset:
```python
model.val(split="test")
```

## Exporting the Model
Once training is complete, export the model in ONNX format:
```python
model.export(format='onnx')
```

## Dataset Structure
Ensure the dataset follows this structure:
```
dataset/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── test/
│   ├── class_1/
│   ├── class_2/
│   └── ...
```

## License
This project is open-source and free to use under the MIT License.

## Authors
Developed by Srimadhav Seebu Kumar and Contributors.


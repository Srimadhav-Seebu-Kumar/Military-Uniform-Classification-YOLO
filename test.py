from ultralytics import YOLO

# Load the trained YOLO classification model
model = YOLO("best.pt")  # Update the path if needed

# Test on a single image
image_path = "images (50).jpg"  # Replace with your test image path
results = model(image_path)

# Print results
print(results)

# Get predicted class
predicted_class = results[0].probs.top1
class_names = model.names  # Get class names from the model
print(f"Predicted class: {class_names[predicted_class]}")

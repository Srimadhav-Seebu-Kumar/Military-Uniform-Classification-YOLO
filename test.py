from ultralytics import YOLO

model = YOLO("best.pt")  

image_path = "images (50).jpg"  
results = model(image_path)

print(results)

predicted_class = results[0].probs.top1
class_names = model.names 
print(f"Predicted class: {class_names[predicted_class]}")

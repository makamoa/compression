import torch
import time
import psutil
import os
from torchvision.ops import box_iou
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Load a sample object detection dataset (e.g., COCO val set)
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.CocoDetection(
    root='./coco/val2017', annFile='./coco/annotations/instances_val2017.json', transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

# Load compressed object detection model (e.g., YOLOv5)
model = torch.load('compressed_object_detection_model.pth')  # Path to compressed model
model.eval()  # Set model to evaluation mode

# Function to calculate IoU-based accuracy for object detection
def calculate_accuracy(outputs, targets, iou_threshold=0.5):
    """ Calculate detection accuracy based on IoU threshold """
    pred_boxes = outputs['boxes']
    target_boxes = targets['boxes']
    
    if len(pred_boxes) == 0 or len(target_boxes) == 0:
        return 0.0
    
    ious = box_iou(pred_boxes, target_boxes)
    iou_matches = ious.max(dim=0).values >= iou_threshold
    accuracy = iou_matches.sum().item() / len(target_boxes)
    return accuracy

# Function to test object detection model performance
def test_object_detection_model(model, test_loader, iou_threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    total_accuracy = 0.0
    total_images = 0
    inference_times = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            
            # Measure inference time
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            
            inference_times.append(end_time - start_time)
            
            # Calculate IoU-based accuracy
            outputs = [{k: v.cpu() for k, v in out.items()} for out in outputs]
            batch_accuracy = calculate_accuracy(outputs[0], targets[0], iou_threshold)
            
            total_accuracy += batch_accuracy
            total_images += 1
    
    avg_accuracy = total_accuracy / total_images * 100
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    print(f'Accuracy (IoU ≥ {iou_threshold}): {avg_accuracy:.2f}%')
    print(f'Average Inference Time per Image: {avg_inference_time:.4f} seconds')
    
    return avg_accuracy, avg_inference_time

# Function to measure model size in MB
def get_model_size(model):
    temp_path = "temp_object_detection_model.pth"
    torch.save(model.state_dict(), temp_path)
    size_in_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)
    return size_in_mb

# Test the object detection model and get performance metrics
print("Testing the compressed object detection model...")
accuracy, avg_inference_time = test_object_detection_model(model, test_loader)

# Get model size
model_size = get_model_size(model)
print(f'Model Size: {model_size:.2f} MB')

# Example baseline values (replace with actual baseline results)
baseline_accuracy = 55.00  # Example baseline accuracy in %
baseline_inference_time = 0.05  # Example baseline inference time in seconds
baseline_model_size = 200.0  # Example baseline model size in MB

# Calculate performance improvements
accuracy_drop = baseline_accuracy - accuracy
speed_improvement = (baseline_inference_time - avg_inference_time) / baseline_inference_time * 100
size_reduction = (baseline_model_size - model_size) / baseline_model_size * 100

print("\nPerformance Comparison:")
print(f'Accuracy Drop: {accuracy_drop:.2f}%')
print(f'Speed Improvement: {speed_improvement:.2f}%')
print(f'Size Reduction: {size_reduction:.2f}%')

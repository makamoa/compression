import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import time
import psutil
import os
from torch.utils.data import DataLoader

# Load CIFAR-100 test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

test_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# Load compressed model (example: ResNet-50)
model = torch.load('compressed_model.pth')  # Path to the saved compressed model
model.eval()  # Set the model to evaluation mode

# Define the criterion (CrossEntropyLoss)
criterion = nn.CrossEntropyLoss()

# Function to measure inference speed, memory usage, and accuracy
def test_model_performance(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    total = 0
    correct = 0
    running_loss = 0.0
    inference_times = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            
            inference_times.append(end_time - start_time)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(test_loader)
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    print(f'Accuracy: {accuracy:.2f}%, Average Loss: {avg_loss:.4f}')
    print(f'Average Inference Time per Batch: {avg_inference_time:.4f} seconds')
    
    return accuracy, avg_loss, avg_inference_time

# Function to measure model size in MB
def get_model_size(model):
    temp_path = "temp.pth"
    torch.save(model.state_dict(), temp_path)
    size_in_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)
    return size_in_mb

# Test the model and get performance metrics
print("Testing the compressed model...")
accuracy, avg_loss, avg_inference_time = test_model_performance(model, test_loader)

# Get model size
model_size = get_model_size(model)
print(f'Model Size: {model_size:.2f} MB')

# Example baseline values (replace with your actual baseline results)
baseline_accuracy = 75.00  # Example baseline accuracy in %
baseline_inference_time = 0.08  # Example baseline inference time in seconds
baseline_model_size = 100.0  # Example baseline model size in MB

# Calculate performance improvements
accuracy_drop = baseline_accuracy - accuracy
speed_improvement = (baseline_inference_time - avg_inference_time) / baseline_inference_time * 100
size_reduction = (baseline_model_size - model_size) / baseline_model_size * 100

print("\nPerformance Comparison:")
print(f'Accuracy Drop: {accuracy_drop:.2f}%')
print(f'Speed Improvement: {speed_improvement:.2f}%')
print(f'Size Reduction: {size_reduction:.2f}%')

# Model Compression Challenge

This document provides the description, objectives, and evaluation pipeline for the **Model Compression Challenge**, focusing on improving speed, memory usage, and maintaining accuracy for image classification and object detection models.

## Challenge Overview
The challenge focuses on compressing models used for two common problems:
1. **Image Classification**
2. **Object Detection**

Participants are required to compress three selected models and improve the following aspects:
- **Inference Speed**: Reduce the average time required for inference.
- **Memory Usage**: Minimize the model size and memory footprint.
- **Accuracy Drop**: Maintain accuracy within an acceptable margin compared to baseline models.

### Selected Models
- **Image Classification**: ResNet-50, MobileNetV2
- **Object Detection**: YOLOv5-S

## Challenge Objectives
1. **Reduce Model Size**: Participants must minimize the number of parameters or overall file size of the model.
2. **Speed Up Inference**: Achieve faster inference on a specified hardware setup.
3. **Maintain Accuracy**: Ensure minimal accuracy drop compared to baseline models.

## Allowed Techniques
Participants can use the following compression techniques:
- **Quantization**: Reducing precision (e.g., 8-bit or mixed-precision)
- **Pruning**: Removing unnecessary weights (structured or unstructured pruning)
- **Knowledge Distillation**: Training a smaller student model using a larger teacher model
- **Neural Architecture Search (NAS)**: Automated search for optimal architectures
- **Hardware-Specific Optimizations**: Using tools like TensorRT or ONNX optimizations

## Datasets
Participants will use standard datasets:
1. **Image Classification**: CIFAR-100, ImageNet
2. **Object Detection**: COCO, Pascal VOC

## Submission Requirements
Each participant must submit:
1. **Compressed model files** (`.pth` or `.onnx`)
2. **Report** detailing:
   - Methods used for compression
   - Speed, memory, and accuracy results
   - Challenges faced and how they were addressed
3. **Code** to reproduce the results

## Evaluation Metrics
The submitted models will be evaluated based on the following metrics:
- **Inference Speed**: Measured as average time per batch or image.
- **Memory Usage**: Measured as the model size in megabytes.
- **Accuracy Drop**: Percentage drop in accuracy or mAP compared to baseline models.

### Scoring
Final ranking will be determined by a weighted score:
- **40% Inference Speed Improvement**
- **30% Model Size Reduction**
- **30% Accuracy Preservation**

Bonus points will be awarded for:
- Using novel or creative compression techniques
- Achieving high performance with minimal computational resources

## Scripts for Model Testing
Two scripts are provided to test compressed models on **CIFAR-100** (image classification) and **COCO** (object detection).

### 1. **test_class.py**
This script tests compressed models on the CIFAR-100 dataset and evaluates:
- **Accuracy**: Percentage of correctly classified images
- **Inference Speed**: Average time per batch
- **Memory Usage**: Model size in megabytes

#### How to Run
```bash
python test_class.py
```
Ensure that the compressed model file is named `compressed_model.pth` and is located in the same directory as the script.

### 2. **test_obj.py**
This script tests compressed object detection models on the COCO dataset and evaluates:
- **IoU-based Accuracy**: Percentage of correctly detected objects with IoU ≥ 0.5
- **Inference Speed**: Average time per image
- **Memory Usage**: Model size in megabytes

#### How to Run
```bash
python test_obj.py
```
Ensure that the compressed object detection model file is named `compressed_object_detection_model.pth` and is located in the same directory as the script.

## Baseline Results
Participants should compare their compressed models against some baseline results (example results):

| Metric                | ResNet-50 | MobileNetV2 | YOLOv5-S |
|-----------------------|-----------|-------------|----------|
| **Accuracy**          | 75.00%    | 71.00%      | 55.00%   |
| **Inference Time**    | 0.10 sec  | 0.05 sec    | 0.05 sec |
| **Model Size**        | 100 MB    | 20 MB       | 200 MB   |

Participants should ensure that their compressed models show significant improvement in speed and size while maintaining accuracy close to these baseline values.

## Contact
For any questions or clarifications, please contact the challenge organizers.

maksim.makarenko @ {aramco.com | kaust.edu.sa}


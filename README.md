Real-Time Road Anomaly Detection using YOLOv5 and ARM-Optimized TensorFlow Lite
🚀 Overview

Road anomalies such as potholes, cracks, and damaged surfaces are major contributors to road accidents and vehicle damage in India.

This project presents an end-to-end Edge AI solution that detects road anomalies in real time using:

YOLOv5 Nano for object detection

INT8 Quantized TensorFlow Lite model

Raspberry Pi 4 (ARM Cortex-A72 CPU)

OpenCV for video processing

The system runs fully offline and is optimized for low-power ARM devices.

🎯 Problem Statement

Manual road inspection is:

Time-consuming

Expensive

Inefficient

Not scalable

Our solution enables:

Automated road monitoring

Real-time anomaly detection

Smart city integration

Low-cost scalable deployment

🧠 Dataset Details

Total Images: 944

Classes: 6

Pothole

Crack

Manhole Cover

Surface Damage

Patch Work

Road Debris

All images were manually labeled using LabelImg/CVAT.

The dataset was designed specifically for Indian road conditions.

🏗 Model Architecture

Base Model: YOLOv5 Nano

Input Resolution: 320x320

Framework: PyTorch

Training Platform: Google Colab (GPU enabled)

📊 Training Results

Strong mAP across all 6 classes

High precision for deep pothole detection

Balanced speed-accuracy tradeoff

(Team can insert actual mAP values here)

⚡ Model Optimization Pipeline

Train YOLOv5 model → best.pt

Export to ONNX

Convert ONNX → TensorFlow

Convert to TFLite

Apply INT8 Quantization

Final Output:

Optimized INT8 TFLite model

~4x size reduction

Faster inference on ARM CPU

💻 Hardware Deployment

Device: Raspberry Pi 4 (8GB RAM)

CPU: ARM Cortex-A72

Camera Module

Active Cooling System

Power Bank (Portable Deployment)

🔥 Performance on Raspberry Pi

FPS: 8–12 FPS

Real-time detection

Offline operation

XNNPACK delegate enabled

📌 System Features

✔ Real-time bounding box detection
✔ Timestamp logging
✔ Low-power deployment
✔ Edge AI architecture
✔ Scalable for smart city integration

🔮 Future Scope

GPS integration

Cloud synchronization

Road damage severity classification

Municipality dashboard integration

📽 Demo Video

[Insert YouTube Link]

👨‍💻 Team Members

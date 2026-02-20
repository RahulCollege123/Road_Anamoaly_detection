Real-Time Road Anomaly Detection On Raspberry Pi 4

Overview
Road anomalies such as potholes, speed breakers, and unexpected obstacles are major contributors to road accidents and vehicle damage in India.
This project presents an end-to-end Edge AI solution that detects road anomalies in real time using:
YOLOv5 Nano for object detection
INT8 Quantized TensorFlow Lite model
Raspberry Pi 4 (ARM Cortex-A72 CPU)
OpenCV for video processing
The system runs fully offline and is optimized for low-power ARM devices.

Problem Statement
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

Dataset Details
Total Images: 944

Classes: 5
Potholes
Speed Breakers
Pedestrians
Animals
Obstacles
All images were manually labeled using Label Sudio
The dataset was designed specifically for Indian road conditions.

Model Architecture:-
Base Model: YOLOv5 Nano
Input Resolution: 256x256
Framework: PyTorch
Training Platform: Google Colab (GPU enabled)

Training Results:-
Strong mAP across all 5 classes
High precision for deep pothole detection
Balanced speed-accuracy tradeoff

                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 8/8 [00:04<00:00,  1.96it/s]
                   all        227        379      0.808      0.751      0.799       0.49
                Animal        227         57      0.931      0.965       0.95      0.716
                 Human        227        180      0.902      0.872      0.927      0.567
              Obstacle        227         72      0.699       0.75       0.76      0.482
               Pothole        227         30      0.729      0.719      0.834      0.478
         Speed Breaker        227         40      0.777       0.45      0.523      0.209
         
Model Optimization Pipeline:-
Train YOLOv5 model → best.pt
Export to ONNX
Convert ONNX → TensorFlow
Convert to TFLite
Apply INT8 Quantization

Final Output:-
Optimized INT8 TFLite model
~4x size reduction
Faster inference on ARM CPU

Hardware Deployment:-
Device: Raspberry Pi 4 (8GB RAM)
CPU: ARM Cortex-A72
Pi Camera Module V2
Active Cooling System
Adapter (5V and 3A)
64GB SD Card SanDisk Ultra
4K HDMI Video Capture Card

Performance on Raspberry Pi 4:-
FPS: 5-8 FPS
Real-time detection
Offline operation
XNNPACK delegate enabled

System Features:-
Real-time bounding box detection
Timestamp logging
Low-power deployment
Edge AI architecture
Scalable for smart city integration

Future Scope
GPS integration
Cloud synchronization
Road damage severity classification
Municipality dashboard integration


🛠️ Setup & Installation
1️⃣ Clone the Repository
git clone https://github.com/RahulCollege123/road_anomaly_detection.git
cd road_anomaly_detection
2️⃣ Create Virtual Environment (Recommended)
Create venv
python3 -m venv edgeai
Activate venv
source edgeai/bin/activate

After activation, you should see:

(edgeai) pi@pi:~/road_anomaly_detection $
3️⃣ Upgrade pip
pip install --upgrade pip
4️⃣ Install Dependencies
pip install numpy opencv-python tflite-runtime

If using Pi Camera (legacy):

sudo apt install python3-opencv

If using Picamera2 (libcamera system):

sudo apt install python3-picamera2

▶️ Running the Project
🟢 1. Live Detection (Camera)

For real-time road anomaly detection:

python detect_live.py

What happens:

Opens Pi Camera / USB Camera

Runs real-time inference

Displays bounding boxes

Logs high-confidence detections (≥ 0.96)

Shows live FPS

Press Q to exit.

🟢 2. Detect From Video File

To run detection on a recorded video:

python detect_video.py

What happens:

Opens file picker

Loads selected video

Runs detection frame-by-frame

Saves output video with bounding boxes

Saves detection log CSV file

Output files are stored in:

outputs/
├── detected_videos/
└── detection_logs/
📊 Detection Logging

Only detections with:

confidence ≥ 0.96

are saved into:

outputs/detection_logs/

Example log:

timestamp_sec,class,confidence
2.34,Pothole,0.97
7.10,Human,0.99
🔥 Optional: Deactivate Virtual Environment

When done:

deactivate

Demo Video
[Insert YouTube Link]

Team Members:-
Deependra Vithharia
Rahul Patil
Saurav Singh

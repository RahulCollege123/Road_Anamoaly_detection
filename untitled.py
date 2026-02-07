import cv2
import time
import numpy as np
import os
from tkinter import Tk, filedialog
import tflite_runtime.interpreter as tflite

# ==============================
# PERFORMANCE SETTINGS
# ==============================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "models/best256-int8.tflite"

OUTPUT_VIDEO_DIR = "outputs/detected_videos"
LOG_DIR = "outputs/detection_logs"

IMG_SIZE = 256
CONF_THRESHOLD = 0.60      # ✅ UPDATED
NMS_THRESHOLD = 0.45

CLASSES = [
    "Animal",
    "Human",
    "Obstacle",
    "Pothole",
    "Speed Breaker",
    "Traffic Cone"
]

os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==============================
# LETTERBOX
# ==============================
def letterbox(img, new_shape=256, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_shape / w, new_shape / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)

    dw = (new_shape - nw) // 2
    dh = (new_shape - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized

    return canvas, scale, dw, dh

# ==============================
# FILE PICKER
# ==============================
root = Tk()
root.withdraw()

video_path = filedialog.askopenfilename(
    title="Select video file",
    filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
)

if not video_path:
    print("❌ No video selected")
    exit()

print(f"📂 Selected video: {video_path}")

# ==============================
# LOAD MODEL
# ==============================
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

out_scale, out_zero = output_details[0]["quantization"]

# ==============================
# VIDEO IO (ORIGINAL SIZE)
# ==============================
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_vid = cap.get(cv2.CAP_PROP_FPS)

name = os.path.splitext(os.path.basename(video_path))[0]

output_video_path = os.path.join(
    OUTPUT_VIDEO_DIR, f"{name}_detected.mp4"
)

log_path = os.path.join(
    LOG_DIR, f"{name}_detections.csv"
)

out = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps_vid,
    (orig_w, orig_h)   # ✅ ORIGINAL RESOLUTION
)

# ==============================
# LOG FILE
# ==============================
log_file = open(log_path, "w")
log_file.write("timestamp_sec,class,confidence\n")

print("🚀 YOLOv5 INT8 detection started")

prev_time = time.time()

# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img, scale, dw, dh = letterbox(frame, IMG_SIZE)
    img = np.expand_dims(img.astype(np.uint8), axis=0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    output_uint8 = interpreter.get_tensor(output_details[0]["index"])[0]
    output = (output_uint8.astype(np.float32) - out_zero) * out_scale

    boxes, scores, class_ids = [], [], []

    for i in range(output.shape[0]):
        x, y, w0, h0 = output[i, 0:4]
        obj_conf = output[i, 4]
        class_scores = output[i, 5:]

        class_id = np.argmax(class_scores)
        confidence = obj_conf * class_scores[class_id]

        if confidence >= CONF_THRESHOLD:
            x1 = (x - w0 / 2) * IMG_SIZE - dw
            y1 = (y - h0 / 2) * IMG_SIZE - dh
            bw = w0 * IMG_SIZE
            bh = h0 * IMG_SIZE

            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            bw = int(bw / scale)
            bh = int(bh / scale)

            boxes.append([x1, y1, bw, bh])
            scores.append(float(confidence))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(
        boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD
    )

    if len(indices) > 0:
        timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            label = CLASSES[class_ids[i]]
            conf = scores[i]

            cv2.rectangle(frame, (x, y), (x + bw, y + bh),
                          (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # ✅ LOG (NO EFFECT ON VIDEO QUALITY)
            log_file.write(
                f"{timestamp_sec:.2f},{label},{conf:.3f}\n"
            )

    curr = time.time()
    fps = 1.0 / (curr - prev_time)
    prev_time = curr

    cv2.putText(frame, f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255), 2)

    out.write(frame)
    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==============================
# CLEANUP
# ==============================
cap.release()
out.release()
log_file.close()
cv2.destroyAllWindows()

print("✅ DONE")
print(f"🎥 Video saved at: {output_video_path}")
print(f"📄 Log saved at: {log_path}")







































import cv2
import time
import numpy as np
import os
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite

# ==============================
# PERFORMANCE SETTINGS
# ==============================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "models/best256-int8.tflite"

IMG_SIZE = 256
CONF_THRESHOLD = 0.60
NMS_THRESHOLD = 0.55

CLASSES = [
    "Animal",
    "Human",
    "Obstacle",
    "Pothole",
    "Speed Breaker",
    "Traffic Cone"
]

# ==============================
# LETTERBOX
# ==============================
def letterbox(img, new_shape=256, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_shape / w, new_shape / h)

    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    dw = (new_shape - nw) // 2
    dh = (new_shape - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized

    return canvas, scale, dw, dh

# ==============================
# LOAD MODEL
# ==============================
interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=2)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

out_scale, out_zero = output_details[0]["quantization"]

print("✅ TFLite model loaded")

# ==============================
# PICAMERA2 INIT (BGR FIX)
# ==============================
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 360), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

print("🎥 Pi Camera started (BGR mode)")

# ==============================
# FPS VARIABLES
# ==============================
prev_time = time.time()
fps_avg = 0.0
frame_id = 0

# ==============================
# MAIN LOOP
# ==============================
while True:
    frame = picam2.capture_array()  # already BGR
    frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

    frame_id += 1

    # 🔥 FPS BOOST: frame skipping
    if frame_id % 2 != 0:
        cv2.imshow("Live Road Anomaly Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # ------------------------------
    # PREPROCESS
    # ------------------------------
    img, scale, dw, dh = letterbox(frame, IMG_SIZE)
    img = np.expand_dims(img, axis=0).astype(np.uint8)

    # ------------------------------
    # INFERENCE
    # ------------------------------
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    output_uint8 = interpreter.get_tensor(output_details[0]["index"])[0]
    output = (output_uint8.astype(np.float32) - out_zero) * out_scale

    boxes, scores, class_ids = [], [], []

    for det in output:
        x, y, w0, h0 = det[:4]
        obj_conf = det[4]
        class_scores = det[5:]

        class_id = np.argmax(class_scores)
        confidence = obj_conf * class_scores[class_id]

        if confidence > CONF_THRESHOLD:
            x1 = (x - w0 / 2) * IMG_SIZE - dw
            y1 = (y - h0 / 2) * IMG_SIZE - dh

            bw = w0 * IMG_SIZE
            bh = h0 * IMG_SIZE

            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            bw = int(bw / scale)
            bh = int(bh / scale)

            boxes.append([x1, y1, bw, bh])
            scores.append(float(confidence))
            class_ids.append(class_id)

    # ------------------------------
    # NMS
    # ------------------------------
    indices = cv2.dnn.NMSBoxes(
        boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD
    )

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            label = CLASSES[class_ids[i]]
            conf = scores[i]

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x, max(y - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # ------------------------------
    # FPS CALC (SMOOTHED)
    # ------------------------------
    curr = time.time()
    fps = 1.0 / (curr - prev_time)
    prev_time = curr
    fps_avg = 0.9 * fps_avg + 0.1 * fps

    cv2.putText(
        frame,
        f"FPS: {fps_avg:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.imshow("Live Road Anomaly Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==============================
# CLEANUP
# ==============================
picam2.stop()
cv2.destroyAllWindows()
print("✅ Camera stopped")

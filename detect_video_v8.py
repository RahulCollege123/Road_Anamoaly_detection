import cv2
import time
import numpy as np
import os
from tkinter import Tk, filedialog
import tflite_runtime.interpreter as tflite

# ==============================
# PERFORMANCE SETTINGS (PI)
# ==============================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "models/bestv8-int8.tflite"

OUTPUT_VIDEO_DIR = "outputs/detected_videos"
LOG_DIR = "outputs/detection_logs"

CONF_THRESHOLD = 0.60
LOG_CONF_THRESHOLD = 0.86
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
def letterbox(img, new_size, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_size / w, new_size / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)

    dw = (new_size - nw) // 2
    dh = (new_size - nh) // 2
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

# INPUT INFO
inp = input_details[0]
_, IMG_H, IMG_W, _ = inp["shape"]
in_dtype = inp["dtype"]
in_scale, in_zero = inp["quantization"]

# OUTPUT INFO
out = output_details[0]
out_scale, out_zero = out["quantization"]

print(f"✅ Model input size: {IMG_W}x{IMG_H}")
print(f"✅ Input dtype: {in_dtype}")
print(f"✅ Input quant: {inp['quantization']}")

# ==============================
# VIDEO IO
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

out_video = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps_vid,
    (orig_w, orig_h)
)

# ==============================
# LOG FILE
# ==============================
log_file = open(log_path, "w")
log_file.write("timestamp_sec,class,confidence\n")

print("🚀 YOLOv8 detection started")

prev_time = time.time()

# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------- PREPROCESS --------
    img, scale, dw, dh = letterbox(frame, IMG_W)

    if in_dtype == np.uint8:
        # TRUE INT8 INPUT
        img = img.astype(np.float32) / 255.0
        img = img / in_scale + in_zero
        img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        # FLOAT32 INPUT (YOLOv8 hybrid INT8)
        img = img.astype(np.float32) / 255.0

    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(inp["index"], img)
    interpreter.invoke()

    # -------- POSTPROCESS --------
    output_raw = interpreter.get_tensor(out["index"])[0]

    if out_scale > 0:
        output = (output_raw.astype(np.float32) - out_zero) * out_scale
    else:
        output = output_raw.astype(np.float32)

    boxes, scores, class_ids = [], [], []

    for det in output:
        x, y, w0, h0 = det[:4]
        obj_conf = det[4]
        class_scores = det[5:]

        class_id = np.argmax(class_scores)
        confidence = obj_conf * class_scores[class_id]

        if confidence >= CONF_THRESHOLD:
            x1 = (x - w0 / 2) * IMG_W - dw
            y1 = (y - h0 / 2) * IMG_H - dh
            bw = w0 * IMG_W
            bh = h0 * IMG_H

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

    # -------- DRAW & LOG --------
    if len(indices) > 0:
        timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            label = CLASSES[class_ids[i]]
            conf = scores[i]

            cv2.rectangle(
                frame, (x, y), (x + bw, y + bh),
                (0, 255, 0), 2
            )

            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )

            if conf >= LOG_CONF_THRESHOLD:
                log_file.write(
                    f"{timestamp_sec:.2f},{label},{conf:.3f}\n"
                )

    # -------- FPS --------
    curr = time.time()
    fps = 1.0 / max(curr - prev_time, 1e-6)
    prev_time = curr

    cv2.putText(
        frame, f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 0, 255), 2
    )

    out_video.write(frame)
    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==============================
# CLEANUP
# ==============================
cap.release()
out_video.release()
log_file.close()
cv2.destroyAllWindows()

print("✅ DONE")
print(f"🎥 Video saved at: {output_video_path}")
print(f"📄 Log saved at: {log_path}")

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
MODEL_PATH = "models/best-int8.tflite"

IMG_SIZE = 256
CONF_THRESHOLD = 0.60        # drawing
LOG_CONF_THRESHOLD = 0.76    # logging only
NMS_THRESHOLD = 0.55

LOG_DIR = "outputs/detection_logs"
os.makedirs(LOG_DIR, exist_ok=True)

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
interpreter = tflite.Interpreter(
    model_path=MODEL_PATH,
    num_threads=2
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

out_scale, out_zero = output_details[0]["quantization"]

print("✅ TFLite model loaded")

# ==============================
# PICAMERA2 INIT
# ==============================
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 360), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

print("🎥 Pi Camera started")

# ==============================
# LOG FILE
# ==============================
log_path = os.path.join(LOG_DIR, "live_camera_detections.csv")
log_file = open(log_path, "w")
log_file.write("timestamp_sec,class,confidence\n")

# ==============================
# FPS VARIABLES
# ==============================
prev_time = time.time()
fps_avg = 0.0
frame_id = 0
start_time = time.time()

# ==============================
# MAIN LOOP
# ==============================
while True:
    frame = picam2.capture_array()
    frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

    frame_id += 1

    # 🔥 frame skip for FPS
    if frame_id % 2 != 0:
        cv2.imshow("Live Road Anomaly Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # ------------------------------
    # PREPROCESS
    # ------------------------------
    img, scale, dw, dh = letterbox(frame, IMG_SIZE)
    img = np.expand_dims(img.astype(np.uint8), axis=0)

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

    # ------------------------------
    # NMS
    # ------------------------------
    indices = cv2.dnn.NMSBoxes(
        boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD
    )

    if len(indices) > 0:
        timestamp_sec = time.time() - start_time

        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            label = CLASSES[class_ids[i]]
            conf = scores[i]

            # draw
            cv2.rectangle(frame, (x, y), (x + bw, y + bh),
                          (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x, max(y - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # ✅ LOG ONLY VERY HIGH CONFIDENCE
            if conf >= LOG_CONF_THRESHOLD:
                log_file.write(
                    f"{timestamp_sec:.2f},{label},{conf:.3f}\n"
                )

    # ------------------------------
    # FPS DISPLAY
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
log_file.close()
picam2.stop()
cv2.destroyAllWindows()
print("✅ Camera stopped")
print(f"📄 Log saved at: {log_path}")

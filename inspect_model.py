import numpy as np
import tflite_runtime.interpreter as tflite

MODEL_PATH = "models/bestv8-int8.tflite"

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# ===============================
# INPUT (MOST IMPORTANT)
# ===============================
inp = interpreter.get_input_details()[0]

input_shape = inp["shape"]          # [1, H, W, 3]
input_dtype = inp["dtype"]
scale, zero = inp["quantization"]

print("\n📥 INPUT")
print("Shape :", input_shape)
print("Dtype :", input_dtype)
print("Scale :", scale)
print("Zero  :", zero)

# ===============================
# OUTPUT (MOST IMPORTANT)
# ===============================
out = interpreter.get_output_details()[0]
output_shape = out["shape"]

print("\n📤 OUTPUT")
print("Shape :", output_shape)

# ===============================
# DERIVED YOLO INFO
# ===============================
H, W = input_shape[1], input_shape[2]

print("\n🧠 YOLO INFO")
print("Input Size :", f"{W}x{H}")

if input_dtype == np.uint8:
    print("Model Type : INT8")
    print("Preprocess : uint8 image → quantized")
elif input_dtype == np.float16:
    print("Model Type : FP16")
    print("Preprocess : image / 255.0")
else:
    print("Model Type : FP32")
    print("Preprocess : image / 255.0")

if len(output_shape) == 3:
    boxes = output_shape[1]
    attrs = output_shape[2]
    classes = attrs - 5
    print("Boxes      :", boxes)
    print("Classes    :", classes)

print("\n✅ Enough info to run model on Raspberry Pi.")

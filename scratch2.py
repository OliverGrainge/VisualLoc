from PlaceRec.Training.models import get_backbone
import torch 
import argparse
import numpy as np
import onnxruntime
import time
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from onnxconverter_common import float16
import onnx

model = get_backbone("resnet101")
model.eval().cuda()
dummy_input = torch.randn(1, 3, 320, 320).cuda()

full_precision_path = "model_fp32.onnx"
half_precision_path = "model_fp16.onnx"
int8_precision_path = "model_int8.onnx"

torch.onnx.export(model,
                  dummy_input,
                  full_precision_path,
                  export_params=True)



def benchmark(model_path, precision):
    providers = [("CUDAExecutionProvider", {"device_id":torch.cuda.current_device(), "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)})]
    sess_options = onnxruntime.SessionOptions()
    if precision == "fp16":
        model = onnx.load(model_path)
        model_fp16 = float16.convert_float_to_float16(model)
        onnx.save(model_fp16, model_path)
    session = onnxruntime.InferenceSession(model_path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
    input_name = session.get_inputs()[0].name
    session.set_providers(['CUDAExecutionProvider'])

    total = 0.0
    runs = 10
    input_data = np.zeros((1, 3, 320, 320), np.float32)
    if precision == "fp16":
        input_data = input_data.astype(np.float16)
    # Warming up
    _ = session.run([], {input_name: input_data})
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")


benchmark(full_precision_path, "fp32")
print("==========================================")
benchmark(full_precision_path, "fp16")
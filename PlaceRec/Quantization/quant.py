import torch_tensorrt
import torch
import torch.nn as nn

def quantize_model(model, precision="fp16", calibration_dataset = None, batch_size=1):
    assert isinstance(model, nn.Module)
    model = model.float().cuda().eval()

    if precision == "fp32":
        trt_model_fp32 = torch_tensorrt.compile(model, inputs = [torch_tensorrt.Input((batch_size, 3, 320, 320), dtype=torch.float32)],
            enabled_precisions = torch.float32, # Run with FP32
            workspace_size = 1 << 22
        )
        return trt_model_fp32
    elif precision == "fp16":  
        trt_model_fp16 = torch_tensorrt.compile(model, inputs = [torch_tensorrt.Input((batch_size, 3, 320, 320), dtype=torch.float32)],
            enabled_precisions = {torch.half, torch.float32}, # Run with FP16
            workspace_size = 1 << 22
        )

        return trt_model_fp16

    elif precision=="int8":
        if calibration_dataset is None:
            raise Exception("must provide calibration dataset for int8 quantization")

        testing_dataloader = torch.utils.data.DataLoader(
            calibration_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )

        calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
            testing_dataloader,
            cache_file="./calibration.cache",
            use_cache=False,
            algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
            #algo_type=torch_tensorrt.ptq.CalibrationAlgo.MINMAX_CALIBRATION,
            device=torch.device("cuda:0"),
        )

        trt_mod = torch_tensorrt.compile(model, inputs=[torch_tensorrt.Input((batch_size, 3, 320, 320))],
                                            enabled_precisions={torch.int8, torch.half, torch.float},
                                            calibrator=calibrator,
                                            truncate_long_and_double=True
                                            #device={
                                            #    "device_type": torch_tensorrt.DeviceType.GPU,
                                            #    "gpu_id": 0,
                                            #    "dla_core": 0,
                                            #    "allow_gpu_fallback": False,
                                            #    "disable_tf32": False
                                            #})
                                            )
        
        return trt_mod
    



def quantize_model_cpu(pytorch_model, precision="fp16", calibration_dataset=None, batch_size=1):
    # Convert PyTorch model to ONNX format
    dummy_input = torch.randn(batch_size, 3, 320, 320)  # Replace 'input_dimensions' with the actual dimensions
    onnx_model_path = "model.onnx"
    torch.onnx.export(pytorch_model, dummy_input, onnx_model_path, opset_version=11)

    # Load the ONNX model
    onnx_model = onnx.load(onnx_model_path)

    # Check and apply quantization based on precision
    if precision == "fp16":
        onnx_model = onnx.numpy_helper.from_array(onnx_model.astype(np.float16))
    elif precision == "int8":
        if calibration_dataset is None:
            raise ValueError("Calibration dataset is required for int8 quantization.")
        onnx_model = quantize_dynamic(onnx_model, per_channel=False, quantization_type=QuantType.QInt8)
    # fp32 doesn't require changes as ONNX default is fp32

    # Save the quantized model (optional)
    quantized_model_path = 'quantized_model.onnx'
    onnx.save(onnx_model, quantized_model_path)

    # Create a session and perform inference
    sess_options = ort.SessionOptions()
    session = ort.InferenceSession(quantized_model_path, sess_options)

    # The rest of the function remains similar to the previous version...

    return session
import torch_tensorrt
import torch
import torch.nn as nn

def quantize_model(model, precision="fp16", calibration_dataset = False, batch_size=1, reload_calib=False, calib_name="calibration.cache"):
    assert isinstance(model, nn.Module)
    model = model.float().cuda().eval()

    if precision == "fp32":
        trt_model_fp32 = torch_tensorrt.compile(model, inputs = [torch_tensorrt.Input((batch_size, 3, 320, 320), dtype=torch.float32)],
            enabled_precisions = torch.float32, # Run with FP32
            workspace_size = 1 << 30
        )
        return trt_model_fp32
    elif precision == "fp16":  
        trt_model_fp16 = torch_tensorrt.compile(model, inputs = [torch_tensorrt.Input((batch_size, 3, 320, 320), dtype=torch.float32)],
            enabled_precisions = {torch.half, torch.float32}, # Run with FP16
            workspace_size = 1 << 30
        )

        return trt_model_fp16

    elif precision=="int8":
        if isinstance(calibration_dataset, bool):
            raise Exception("must pass a dataloader for int8 quantization")
        
        testing_dataloader = torch.utils.data.DataLoader(
            calibration_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
        print("./calibration_cache/" + calib_name + ".cache")
        calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
            testing_dataloader,
            cache_file="./calibration_cache/" + calib_name + ".cache",
            use_cache=reload_calib,
            algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
            device=torch.device("cuda:0"),
        )

        trt_mod = torch_tensorrt.compile(model, inputs=[torch_tensorrt.Input((batch_size, 3, 320, 320))],
                                            enabled_precisions={torch.int8, torch.half, torch.float},
                                            calibrator=calibrator,
                                            truncate_long_and_double=True, 
                                            workspace_size = 1 << 30,
                                            #device={
                                            #    "device_type": torch_tensorrt.DeviceType.GPU,
                                            #    "gpu_id": 0,
                                            #    "dla_core": 0,
                                            #    "allow_gpu_fallback": False,
                                            #    "disable_tf32": False
                                            #})
                                            )
        

        return trt_mod
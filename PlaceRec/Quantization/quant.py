import torch_tensorrt
import torch
import torch.nn as nn
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from torch.utils.data import DataLoader, Subset
import os
from PlaceRec.Training.dataloaders.val.MapillaryDataset import MSLS
import torchvision.transforms as T

def quantize_model(model, precision="fp16", calibration_dataset = False, batch_size=1, reload_calib=False, calib_name="calibration.cache"):
    assert isinstance(model, torch.nn.Module)
    model = model.float().cuda().eval()

    if precision == "fp32":
        trt_model_fp32 = torch_tensorrt.compile(model, inputs = [torch_tensorrt.Input((batch_size, 3, 320, 320), dtype=torch.float32)],
            enabled_precisions = torch.float32, # Run with FP32
            workspace_size = 2 << 30
        )
        return trt_model_fp32
    elif precision == "fp16":  
        trt_model_fp16 = torch_tensorrt.compile(model, inputs = [torch_tensorrt.Input((batch_size, 3, 320, 320), dtype=torch.float32)],
            enabled_precisions = {torch.half, torch.float32}, # Run with FP16
            workspace_size = 2 << 30
        )

        return trt_model_fp16

    elif precision=="int8":
        if isinstance(calibration_dataset, bool):
            raise Exception("must pass a dataloader for int8 quantization")
        
        testing_dataloader = torch.utils.data.DataLoader(
            calibration_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
        calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
            testing_dataloader,
            cache_file="./calibration_cache/" + calib_name + ".cache",
            use_cache=reload_calib,
            algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
            device=torch.device("cuda:0"),
        )

        trt_mod = torch_tensorrt.compile(model, inputs=[torch_tensorrt.Input((batch_size, 3, 320, 320), dtype=torch.float32)],
                                            enabled_precisions={torch.int8},
                                            calibrator=calibrator,
                                            truncate_long_and_double=True, 
                                            workspace_size = 2 << 30,
                                            #device={
                                            #    "device_type": torch_tensorrt.DeviceType.GPU,
                                            #    "gpu_id": 0,
                                            #    "dla_core": 0,
                                            #    "allow_gpu_fallback": False,
                                            #    "disable_tf32": False
                                            #})
                                            )
        

        return trt_mod
    



IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}

valid_transform = T.Compose([
            T.Resize((320, 320), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"])])


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, model_name):
        trt.IInt8EntropyCalibrator2.__init__(self)
        calibration_ds = MSLS(valid_transform)
        calibration_ds = Subset(calibration_ds, list(range(100)))
        calibration_dataloader = DataLoader(calibration_ds, batch_size=1, num_workers=0)
        self.calibration_dataloader = iter(calibration_dataloader)  # Make it an iterator
        self.calibration_cache_path = "./calibration_cache/" + model_name + ".cache"
        self.current_batch = None
        self.cache_data = None
        # Assuming the dataloader is set up to return a batch of images as the first element
        # and potentially targets as the second, we'll grab the first batch to allocate memory.
        first_batch, _ = next(self.calibration_dataloader)
        _, C, H, W = first_batch.shape
        self.device_input = cuda.mem_alloc(first_batch.numpy().nbytes)
        self.batch_size = first_batch.size(0)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        try:
            # Fetch the next batch of images
            batch, _ = next(self.calibration_dataloader)
            if batch is None:
                return None
            # Convert PyTorch tensor to numpy and ensure memory is contiguous
            batch = batch.numpy().ravel()  # Flatten to ensure contiguous memory
            cuda.memcpy_htod(self.device_input, batch)
            return [int(self.device_input)]
        except StopIteration:
            # No more data to process
            return None

    def read_calibration_cache(self):
        try:
            with open(self.calibration_cache_path, "rb") as f:
                self.cache_data = f.read()
                return self.cache_data
        except:
            return None

    def write_calibration_cache(self, cache):
        with open(self.calibration_cache_path, "wb") as f:
            f.write(cache)


def build_engine_onnx_int8(model_file, calibrator, force_recalibration=True): 
    if force_recalibration and os.path.exists(calibrator.calibration_cache_path):
        os.remove(calibrator.calibration_cache_path)
        print("Deleted existing calibration cache to force recalibration.")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        

        
        builder.max_batch_size = calibrator.get_batch_size()
        #builder.int8_mode = True
        
        config = builder.create_builder_config()
        config.max_workspace_size = 2 << 30
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator
        #config.set_flat(trt.BuilderFlag.INT8)

        with open(model_file, 'rb') as onnx_model:
            if not parser.parse(onnx_model.read()):
                print("Failed to parse ONNX model. Please check your model file.")
                return None
        
        # Build the TensorRT engine
        with builder.build_engine(network, config) as engine:
            if engine is None:
                print("Failed to build TensorRT engine. Please check your model and calibration data.")
                return None
            
            # Serialize the TensorRT engine
            serialized_engine = engine.serialize()
            return serialized_engine



def build_engine_onnx_fp(model_file, precision):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30 # 1GB 
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "fp32":
        config.set_flag(trt.BuilderFlag.FP16)
    else: 
        raise Exception("precision must be fp16 or fp32")
    # Load ONNX model
    with open(model_file, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    return builder.build_serialized_network(network, config)

    

def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

def get_infer(engine, descriptor_size=1024):
    def infer(input_data):
        # Allocate buffers for input and output
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.detach().cpu().numpy()

        h_input = np.array(input_data, dtype=np.float32)
        h_output = np.empty([descriptor_size], dtype=np.float32) # Example output size, adjust based on model
        d_input = cuda.mem_alloc(1 * h_input.nbytes)
        d_output = cuda.mem_alloc(1 * h_output.nbytes)
        bindings = [int(d_input), int(d_output)]
        stream = cuda.Stream()
        # Create a context for the engine
        context = engine.create_execution_context()
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input, h_input, stream)
        # Execute the model
        context.execute_async_v2(bindings, stream.handle, None)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        return torch.tensor([h_output]).float()
    return infer




def quantize_model_trt(model, precision="fp16", force_recalibration=False, model_name="calibration", descriptor_size=1024):
    x = torch.randn(1, 3, 320, 320).cuda()
    assert isinstance(model, nn.Module)
    model.eval().cuda()
    torch.onnx.export(model, x, "model_fp32.onnx", verbose=True, input_names=["input"], output_names=['output'])
    if precision in ["fp16", "fp32"]:
        engine = build_engine_onnx_fp("model_fp32.onnx", precision=precision)
    elif precision == "int8":
        calibrator = EntropyCalibrator(model_name)
        engine = build_engine_onnx_int8("model_fp32.onnx", calibrator=calibrator, force_recalibration=force_recalibration)
    else: 
        raise Exception("Precision must be int8, fp16 or fp32")
    
    with open("model_fp32.trt", "wb") as f:
        f.write(engine)
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(TRT_LOGGER)
    engine = load_engine(trt_runtime, "model_fp32.trt")
    forward = get_infer(engine, descriptor_size=descriptor_size)
    return forward

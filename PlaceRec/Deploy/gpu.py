import numpy as np
import torch
import torch_tensorrt
import tensorrt as trt
from typing import List
import pycuda.driver as cuda
import pycuda.autoinit
import onnx
from onnx import numpy_helper


def load_engine(trt_runtime, engine_path):
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


def get_infer(engine, descriptor_size=1024):
    def infer(input_data):
        # Allocate buffers for input and output
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.detach().cpu().numpy()

        h_input = np.array(input_data, dtype=np.float32)
        h_output = np.empty(
            descriptor_size, dtype=np.float32
        )  # Example output size, adjust based on model
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
        return torch.tensor(np.array([h_output])).float()

    return infer


def deploy_gpu(method, batch_size=1, precision: List = ["fp16", "fp32"], sparse=False):
    if not torch.cuda.is_available():
        Exception("Cannot deploy model on gpu and cuda is not found")
    method.set_device("cuda")
    model = method.model
    model.eval()
    img = method.example_input().cuda().repeat(batch_size, 1, 1, 1)

    torch.onnx.export(
        model,
        img,
        ".model.onnx",
        verbose=False,
        input_names=["input"],
        output_names=["output"],
    )

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:

        builder.max_batch_size = batch_size
        # builder.int8_mode = True

        config = builder.create_builder_config()
        # config.max_workspace_size = 2 << 30

        if sparse:
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
            config.set_flag(trt.BuilderFlag.FP16)
            # config.set_flag(trt.BuilderFlag.INT8)

        else:
            if "fp16" in precision:
                config.set_flag(trt.BuilderFlag.FP16)
            if "int8" in precision:
                config.set_flag(trt.BuilderFlag.INT8)
        # config.int8_calibrator = calibrator

        with open(".model.onnx", "rb") as onnx_model:
            if not parser.parse(onnx_model.read()):
                print("Failed to parse ONNX model. Please check your model file.")
                return None

        # Build the TensorRT engine
        with builder.build_engine(network, config) as engine:
            if engine is None:
                print(
                    "Failed to build TensorRT engine. Please check your model and calibration data."
                )
                return None

            # Serialize the TensorRT engine
            serialized_engine = engine.serialize()
            engine_path = ".model.trt"

            with open(engine_path, "wb") as f:
                f.write(serialized_engine)

        trt_runtime = trt.Runtime(TRT_LOGGER)
        engine = load_engine(trt_runtime, engine_path)
        print(
            "================================================================",
            method.features_dim["global_feature_shape"],
        )
        method.inference = get_infer(
            engine, method.features_dim["global_feature_shape"]
        )
        return method


def deploy_tensorrt_sparse(method):
    if not torch.cuda.is_available():
        raise Exception("GPU is not visible. Cannot compile to tensorrt backend")

    method.set_device("cuda")
    method.model.eval()
    dummy_input = method.example_input().cuda()
    method.model(dummy_input)
    method.model = torch.jit.trace(method.model, dummy_input)
    compiled_model = torch_tensorrt.compile(
        method.model,
        inputs=[torch_tensorrt.Input(dummy_input.shape)],
        enabled_precisions={torch.float},  # Specify precision
        sparse_weights=True,
        truncate_long_and_double=True,
        # require_full_compilation=True
    )  # Enable sparsity

    compiled_model(dummy_input)

    def inference(x):
        out = compiled_model(x)
        return out[0]

    method.inference = inference
    return method


def deploy_tensorrt(method):
    if not torch.cuda.is_available():
        raise Exception("GPU is not visible. Cannot compile to tensorrt backend")

    method.set_device("cuda")
    method.model.eval()
    dummy_input = method.example_input().cuda()
    method.model(dummy_input)
    method.model = torch.jit.trace(method.model, dummy_input)
    compiled_model = torch_tensorrt.compile(
        method.model,
        inputs=[torch_tensorrt.Input(dummy_input.shape)],
        enabled_precisions={torch.float},  # Specify precision
        sparse_weights=False,
        truncate_long_and_double=True,
        # require_full_compilation=True
    )  # Enable sparsity

    compiled_model(dummy_input)

    def inference(x):
        out = compiled_model(x)
        return out[0]

    method.inference = inference
    return method

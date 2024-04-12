import numpy as np
import pycuda.autoinit  # Automatically initializes CUDA
import pycuda.driver as cuda
import torch
import torch_tensorrt


def deploy_tensorrt_sparse(method):
    if not torch.cuda.is_available():
        raise Exception("GPU is not visible. Cannot compile to tensorrt backend")

    method.set_device("cuda")
    method.model.eval()
    dummy_input = method.example_input().cuda()
    method.model(dummy_input)
    compiled_model = torch_tensorrt.compile(
        method.model,
        inputs=[torch_tensorrt.Input(dummy_input.shape)],
        enabled_precisions={torch.float},  # Specify precision
        enable_sparsity=True,
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
    compiled_model = torch_tensorrt.compile(
        method.model,
        inputs=[torch_tensorrt.Input(dummy_input.shape)],
        enabled_precisions={torch.float},  # Specify precision
        enable_sparsity=False,
    )  # Enable sparsity

    compiled_model(dummy_input)

    def inference(x):
        out = compiled_model(x)
        return out[0]

    method.inference = inference
    return method

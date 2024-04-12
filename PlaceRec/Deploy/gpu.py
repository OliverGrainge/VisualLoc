import numpy as np
import torch
import torch_tensorrt


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

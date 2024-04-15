import numpy as np
import onnx
import onnxruntime as ort
import torch
from deepsparse import compile_model


def deploy_cpu(method, sparse=False):
    method.set_device("cpu")
    model = method.model
    model.eval()
    img = method.example_input()

    torch.onnx.export(
        model,
        img,
        ".model.onnx",
        verbose=False,
        input_names=["input"],
        output_names=["output"],
        # dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    onnx_model = onnx.load(".model.onnx")
    onnx.checker.check_model(onnx_model)

    if not sparse:
        session = ort.InferenceSession(".model.onnx")

        def inference(x):
            x = x.detach().cpu().numpy()
            x = list(x)
            outputs = session.run(["output"], {"input": x})
            outputs = torch.stack([torch.from_numpy(y) for y in outputs])
            return outputs

    else:
        engine = compile_model(".model.onnx", batch_size=1, scheduler="single_stream")
        inputs = [img.numpy()]
        engine.run(inputs)

        def inference(x):
            x = x.detach().cpu().numpy()
            x = list(x[None, :])
            outputs = engine.run(x)
            return outputs

    method.inference = inference
    return method

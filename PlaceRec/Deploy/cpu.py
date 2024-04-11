import numpy as np
import onnx
import onnxruntime as ort
import torch


def deploy_onnx_cpu(method):
    method.set_device("cpu")
    model = method.model
    model.eval()
    img = np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8)
    img = method.preprocess(img)[None, :]
    torch.onnx.export(
        model,
        img,
        ".model.onnx",
        verbose=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    onnx_model = onnx.load(".model.onnx")
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(".model.onnx")

    def inference(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        x = list(x)
        outputs = session.run(["output"], {"input": x})
        outputs = torch.stack([torch.from_numpy(y) for y in outputs])
        return outputs

    method.inference = inference
    return method


def deploy_sparse_cpu(method):
    from deepsparse import compile_model
    from deepsparse.utils import generate_random_inputs

    model = method.model
    model.eval()
    img = np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8)
    img = method.preprocess(img)[None, :]
    torch.onnx.export(
        model,
        img,
        ".model.onnx",
        verbose=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    engine = compile_model(".model.onnx", batch_size=1)
    inputs = generate_random_inputs(engine.input_metadata)
    engine.run(inputs)

    def inference(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = list(x)
        outputs = engine.run(x)
        return outputs

    method.inference = inference
    return method

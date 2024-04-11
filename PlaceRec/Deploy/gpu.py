import numpy as np
import pycuda.autoinit  # Automatically initializes CUDA
import pycuda.driver as cuda
import tensorrt as trt
import torch


def deploy_tensorrt_sparse(method):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Load the ONNX model and convert it to a TensorRT engine
    def build_engine(onnx_file_path):
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH
        ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:

            builder.max_workspace_size = 1 << 30  # Adjust size as needed
            builder.max_batch_size = 1
            builder.fp16_mode = (
                True  # Enable FP16 if supported by your GPU to improve performance
            )
            # Load the ONNX model into the parser
            with open(onnx_file_path, "rb") as model:
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

            # The following optimizations are where you can tweak for sparsity and other optimizations
            # builder.setFlag(trt.BuilderFlag.SPARSE_WEIGHTS)  # Uncomment if TensorRT version supports it

            return builder.build_cuda_engine(network)

    model = method.model
    model.eval()
    img = np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8)
    img = method.preprocess(img)[None, :]

    # Export the model to ONNX
    torch.onnx.export(
        model,
        img,
        ".model.onnx",
        verbose=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Build the TensorRT engine
    engine = build_engine(".model.onnx")
    if engine is None:
        print("Engine creation failed")
        return method

    # Allocate buffers and create a CUDA stream
    import pycuda.driver as cuda

    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    def inference(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        # The data x needs to be copied to the input buffer
        # Here we assume x is already preprocessed and in the correct shape
        np.copyto(inputs[0].host, x.ravel())

        [output] = do_inference_v2(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )
        # Convert output to a format your method expects (e.g., torch tensor)
        output_tensor = torch.from_numpy(output)
        return output_tensor

    method.inference = inference
    return method

import os
import onnxruntime as ort
import numpy as np


def load_and_infer_onnx_models(directory, input_tensor):
    # List all files in the directory
    model_files = [f for f in os.listdir(directory) if f.endswith(".onnx")]

    # Loop through each model file
    for model_file in model_files:
        model_path = os.path.join(directory, model_file)
        print(f"Loading model from {model_path}")

        # Set up the ONNX Runtime inference session
        sess = ort.InferenceSession(model_path)

        # Assuming the model has one input and one output
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        # Run inference
        result = sess.run([output_name], {input_name: input_tensor})

        # Print the result
        print(f"Inference result for {model_file}: {result}")

        # Define the directory containing the ONNX models


directory = "/Users/olivergrainge/Downloads/Onnx_Checkpoints"

# Create a dummy input tensor that matches the expected input shape of your models
# Example: a tensor with shape (1, 3, 224, 224) for models expecting 224x224 RGB images
dummy_input = np.random.randn(1, 3, 320, 320).astype(np.float32)

# Run the function to load models and perform inference
load_and_infer_onnx_models(directory, dummy_input)

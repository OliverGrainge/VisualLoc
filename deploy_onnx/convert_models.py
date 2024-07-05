import torch
import os
from pathlib import Path
from PlaceRec.Methods import ResNet50_GeM
from PlaceRec.utils import get_method
from PlaceRec.Methods.gem import GeM

# Directory containing the PyTorch model files
source_directory = "/Users/olivergrainge/Downloads/Checkpoints"
# Directory where ONNX models will be saved
destination_directory = "/Users/olivergrainge/Downloads/Onnx_Checkpoints"

# Create the destination directory if it does not exist
Path(destination_directory).mkdir(parents=True, exist_ok=True)

# Load each model and export it as an ONNX file
for model_file in os.listdir(source_directory):
    if model_file.endswith(".ckpt") and "NetVLAD" in model_file:
        # Construct the full path to the model file
        model_path = os.path.join(source_directory, model_file)
        # Load the PyTorch model
        model = torch.load(model_path, map_location="cpu")

        model.eval()  # Set the model to evaluation mode

        # Dummy input for the export; adjust the size as per model requirement
        dummy_input = torch.randn(1, 3, 320, 320)  # Example for an image input model

        # out = model(dummy_input)
        # print("=-=============================", out.shape)
        # raise Exception

        # Construct the path to save the ONNX model
        onnx_model_path = os.path.join(
            destination_directory, model_file.replace(".ckpt", ".onnx")
        )

        # Export the model to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_model_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        print(f"Exported {model_file} to {onnx_model_path}")

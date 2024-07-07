import torch
import os
from pathlib import Path
from PlaceRec.Methods import ResNet50_GeM
from PlaceRec.utils import get_method
from PlaceRec.Methods.gem import GeM

# Directory containing the PyTorch model files
source_directory = "/Users/olivergrainge/Downloads/Checkpoints"
# Directory where ONNX models will be saved
destination_directory = "/Users/olivergrainge/Downloads/ResNet34_Onnx_Checkpoints"

# Create the destination directory if it does not exist
Path(destination_directory).mkdir(parents=True, exist_ok=True)

# Load each model and export it as an ONNX file
for model_file in os.listdir(source_directory):
    # Construct the full path to the model file

    model_path = os.path.join(source_directory, model_file)
    print("===")
    print("===")
    print("===")
    print("==============", model_path)
    print("===")
    print("===")
    print("===")
    if ".ckpt" not in model_path:
        continue
    # Load the PyTorch model
    model = torch.load(model_path, map_location="cpu")

    model.eval()  # Set the model to evaluation mode

    # Dummy input for the export; adjust the size as per model requirement
    dummy_input = torch.randn(1, 3, 320, 320)  # Example for an image input model

    # Construct the path to save the ONNX model
    onnx_model_path = os.path.join(
        destination_directory, model_file.replace(".ckpt", ".onnx")
    )

    # Export the model to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
    )

    print(f"Exported {model_file} to {onnx_model_path}")

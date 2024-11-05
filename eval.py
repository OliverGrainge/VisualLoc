import os
import pickle

from parsers import eval_arguments
from PlaceRec.Evaluate import Eval
from PlaceRec.utils import get_dataset, get_method

args = eval_arguments()


def load_result(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        return {}


def save_result(file_path: str, result: dict) -> dict:
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(result, f)


for method_name in args.methods:
    method = get_method(method_name, pretrained=True)
    for dataset_name in args.datasets:
        dataset = get_dataset(dataset_name)
        eval = Eval(method, dataset)
        eval.eval()
        #new_result = eval.ratk(1)
        #result_path = f"PlaceRec/Evaluate/results/{method.name}.pkl"
        #old_result = load_result(result_path)
        #new_result = old_result | new_result
        #save_result(result_path, new_result)

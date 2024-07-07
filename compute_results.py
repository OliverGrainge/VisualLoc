import PlaceRec.Methods as Methods
import PlaceRec.Datasets as Datasets
from PlaceRec.utils import get_config, get_method, get_dataset
from PlaceRec.Evaluate import Eval
import pickle
import os
import pandas as pd
import torch

type = "latency"  # either accuracy or latency.
datasets = ["MapillarySLS", "Pitts30k_Val", "AmsterTime", "SVOX"]
directory = "/home/oliver/Documents/github/ResNet34_Onnx_Checkpoints"
# directory = "/home/oliver/Documents/github/ResNet34_Checkpoints"
methods = ["ResNet34_MixVPR", "ResNet34_GeM", "ResNet34_NetVLAD", "ResNet34_ConvAP"]
gammas = [0.00, 0.25, 0.50, 0.75]

config = get_config()


def load_model_weights(method_name, aggregation_pruning_rate):
    # Path to the directory containing the checkpoint files
    filenames = os.listdir(directory)
    filtered_files = []
    for filename in filenames:
        if f"{method_name}_agg_{aggregation_pruning_rate}" in filename:
            filtered_files.append(filename)
    sorted_files = sorted(filtered_files, key=lambda x: float(x.split("_")[5]))
    return sorted_files


weights = load_model_weights("ConvAP", 0.25)


def compute_descriptors(weight_pth):
    method_name = weight_pth.split("_")[0:2]
    method_name = "_".join(method_name)
    method = get_method(method_name, pretrained=False)
    method.model = torch.load(os.path.join(directory, weight_pth), map_location="cpu")
    # method.load_weights(os.path.join(directory, weight_pth))
    method.features_dim = method.features_size()
    method.set_device(config["run"]["device"])
    for dataset_name in datasets:
        ds = get_dataset(dataset_name)
        map_loader = ds.map_images_loader(
            preprocess=method.preprocess,
            num_workers=config["run"]["num_workers"],
            pin_memory=config["run"]["pin_memory"],
            batch_size=config["run"]["batchsize"],
        )
        _ = method.compute_map_desc(dataloader=map_loader)
        del map_loader
        query_loader = ds.query_images_loader(
            preprocess=method.preprocess,
            num_workers=config["run"]["num_workers"],
            pin_memory=config["run"]["pin_memory"],
            batch_size=config["run"]["batchsize"],
        )
        _ = method.compute_query_desc(dataloader=query_loader)
        del query_loader
        method.save_descriptors(ds.name)
        del ds


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


def load_results():
    try:
        results = pd.read_csv("results.csv")
        results.set_index("weight_path", inplace=True)
    except:
        basic_columns = [
            "weight_path",
            "method_name",
            "sparsity",
            "descriptor_dim",
            "flops",
            "model_memory",
            "extraction_lat_cpu_bs1",
            "extraction_lat_gpu_bs1",
        ]

        for dataset in datasets:
            basic_columns += [
                f"{dataset}_R1",
                f"{dataset}_R5",
                f"{dataset}_R10",
                f"{dataset}_map_memory",
                f"{dataset}_total_memory",
                f"{dataset}_matching_lat",
                f"{dataset}_total_cpu_lat_bs1",
                f"{dataset}_total_gpu_lat_bs1",
            ]

        results = pd.DataFrame(columns=basic_columns)
        results.set_index("weight_path", inplace=True)
    return results


def compute_recalls(weight_pth, results):
    method_name = weight_pth.split("_")[0:2]
    method_name = "_".join(method_name)

    if ".ckpt" in weight_pth:
        method = get_method(method_name, pretrained=False)
        method.model = torch.load(
            os.path.join(directory, weight_pth), map_location="cpu"
        )
        method.features_dim = method.features_size()
    else:
        method = get_method(method_name, pretrained=False)

    run_once = False
    for dataset_name in datasets:
        if ".onnx" in weight_pth:
            eval = Eval(
                method, "Pitts30k_Val", onnx_pth=os.path.join(directory, weight_pth)
            )
        else:
            dataset = get_dataset(dataset_name)
            eval = Eval(method, dataset)
            descriptor_dim = eval.descriptor_dim()
            results.loc[weight_pth, "descriptor_dim"] = descriptor_dim

        if run_once == False:
            if ".onnx" in weight_pth:
                pth = weight_pth.replace(".onnx", ".ckpt")
            else:
                pth = weight_pth
            flops = eval.count_flops()
            model_memory = eval.model_memory()
            sparsity = float(pth.split("_")[5])
            results.loc[pth, "method_name"] = method_name
            results.loc[pth, "flops"] = flops
            results.loc[pth, "model_memory"] = model_memory
            results.loc[pth, "sparsity"] = sparsity
            run_once = True

        if type == "accuracy":
            eval.compute_all_matches()
            rat1 = eval.ratk(1)
            rat5 = eval.ratk(5)
            rat10 = eval.ratk(10)
            map_memory = eval.map_memory()
            total_memory = map_memory + model_memory

            results.loc[weight_pth, f"{dataset_name}_R1"] = rat1
            results.loc[weight_pth, f"{dataset_name}_R5"] = rat5
            results.loc[weight_pth, f"{dataset_name}_R10"] = rat10
            results.loc[weight_pth, f"{dataset_name}_map_memory"] = map_memory
            results.loc[weight_pth, f"{dataset_name}_total_memory"] = total_memory

        else:

            cpu_lat_bs1 = eval.extraction_cpu_latency()
            gpu_lat_bs1 = eval.extraction_gpu_latency()
            mat_lat = eval.matching_latency()
            print("======", mat_lat)
            cpu_total_lat_bs1 = cpu_lat_bs1 + mat_lat
            gpu_total_lat_bs1 = gpu_lat_bs1 + mat_lat

            pth = weight_pth.replace(".onnx", ".ckpt")
            print("============================================", pth)
            results.loc[pth, f"{dataset_name}_matching_lat"] = mat_lat
            results.loc[pth, "extraction_lat_cpu_bs1"] = cpu_lat_bs1
            results.loc[pth, "extraction_lat_gpu_bs1"] = gpu_lat_bs1

            results.loc[pth, f"{dataset_name}_total_gpu_lat_bs1"] = gpu_total_lat_bs1
            results.loc[pth, f"{dataset_name}_total_cpu_lat_bs1"] = cpu_total_lat_bs1

    results.to_csv("results.csv")


weights_pths = load_model_weights("ResNet34_ConvAP", 0.75)
for pth in weights_pths:
    results = load_results()
    if type == "accuracy":
        compute_descriptors(pth)
    print(
        "========================================================================++++",
        pth,
    )
    compute_recalls(pth, results)


"""
for method in methods: 
    for gamma in gammas:
        weights_pths = load_model_weights(method, gamma)
        for pth in weights_pths:
            results = load_results()
            if type == "accuracy":
                compute_descriptors(pth)
            compute_recalls(pth, results)
"""

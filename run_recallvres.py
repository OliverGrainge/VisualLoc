
import yaml

from PlaceRec.utils import get_config, get_dataset, get_method
from PlaceRec.Metrics import recallatk
from torchvision import transforms
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import pickle
import os
from os.path import join
from glob import glob
import torch
from parsers import train_args
from thop import profile
from PIL import Image

config = get_config()
args = train_args()


datasets = ["spedtest"]
teacher_method_name = "resnet50convap"
student_methods = ["resnet34convap", "resnet34convap"]
training_type = ["uniform_random_resize", "single_resolution"]
weight_paths = [
    "/home/oliver/Documents/github/VisualLoc/Distillation/checkpoints/resnet50convap/resnet34convap_uniform_random_resize_480-640-epoch=03-val_loss=0.00015.ckpt",
    "Distillation/checkpoints/resnet50convap/resnet34convap_single_resolution_480-640-epoch=02-val_loss=0.00013.ckpt"]
n_points = 10


def replace_resize_transform(pipeline: transforms.Compose, size: Tuple[int, int]):
    """
    Replaces the Resize transform in a torchvision transformation pipeline with a new transform.

    :param pipeline: The original transformation pipeline (transforms.Compose object).
    :param new_transform: The new transformation to replace Resize with.
    :return: A new transformation pipeline with Resize replaced by new_transform.
    """
    new_pipeline = []
    for transform in pipeline.transforms:
        if isinstance(transform, transforms.Resize):
            new_pipeline.append(transforms.Resize(size, antialias=True))
        else:
            new_pipeline.append(transform)
    return transforms.Compose(new_pipeline)


def get_resize_size(pipeline: transforms.Compose) -> Tuple[int, int]:
    for transform in pipeline.transforms:
        if isinstance(transform, transforms.Resize):
            return transform.size
    raise Exception("Pipeline has no Resize")


def plot_curve(results: dict, dataset_name):
    fig, ax = plt.subplots()
    method_names = list(results.keys())
    for method_name in method_names:
        recalls = results[method_name][1]
        resolutions = results[method_name][0]
        plt.plot(resolutions, recalls, label=method_name)
    plt.title(dataset_name)
    plt.legend()
    pth = os.getcwd() + "/Plots/PlaceRec/RecallvResolution"
    if not os.path.exists(pth):
        os.makedirs(pth)
    fig.savefig(pth + "/" + dataset_name + ".png")
    return ax


#################################### Computing Descriptors ###########################################

results = {}
for dataset_name in datasets:
    ds = get_dataset(dataset_name)
    results[dataset_name] = {}
    for i, method_name in enumerate(student_methods):
        print("========================================================", method_name, "=================================================================")
        student_method = get_method(method_name, pretrained=True) #weights=None
        teacher_method = get_method(teacher_method_name, pretrained=True) #weights=None
        print(weight_paths[i])
        if os.path.exists(weight_paths[i]): 
            state_dict = torch.load(weight_paths[i])["state_dict"]
            new_state_dict = {}
            for key in list(state_dict.keys()):
                new_state_dict[key.replace("model.backbone.model", "backbone.model").replace("model.aggregator", "aggregator")] = state_dict[key]
            student_method.model.load_state_dict(new_state_dict)
        else: 
            print("Using Default Pretrained Weights")

        (h, w) = get_resize_size(student_method.preprocess)
        multipliers = np.linspace(0.2, 1.0, n_points)
        resolutions = [(int(mult*h), int(mult*w)) for mult in multipliers]

        map_loader = ds.map_images_loader(
            preprocess=student_method.preprocess,
            num_workers=args.num_workers,
            pin_memory=False)
        map_desc = student_method.compute_map_desc(dataloader=map_loader)
        del map_loader
        rec, flops, res = [], [], []
        for idx, size in enumerate(resolutions):
            preprocess = replace_resize_transform(student_method.preprocess, size)
            sample_image = Image.fromarray(np.random.randint(0, 255, (480, 640, 3)).astype(np.uint8))
            sample_image = preprocess(sample_image).cuda()
            fl, _ = profile(student_method.model, inputs=(sample_image[None, :], ))
            query_loader = ds.query_images_loader(
                preprocess=preprocess,
                num_workers=args.num_workers,
                pin_memory=False,
            )
            _ = student_method.compute_query_desc(dataloader=query_loader)
            del query_loader
            student_method.set_map(map_desc)
            rec.append(recallatk(student_method, ds.ground_truth(), k=1))
            res.append(multipliers[idx])
            flops.append(fl)
        del student_method
        results[dataset_name][method_name + "_" + training_type[i]] = [res, flops, rec]
    with open(join('Data/', dataset_name + '.pkl'), 'wb') as file:
        pickle.dump(results[dataset_name], file)
    del ds
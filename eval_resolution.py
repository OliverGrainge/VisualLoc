import argparse

import yaml

from parsers import eval_resolution_arguments
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

config = get_config()
args = eval_resolution_arguments()



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

if args.mode == "run":
    results = {}
    for dataset_name in args.datasets:
        ds = get_dataset(dataset_name)
        results[dataset_name] = {}
        for i, method_name in enumerate(args.student_methods):
            student_method = get_method(method_name, pretrained=True) #weights=None
            teacher_method = get_method(args.teacher_method, pretrained=True) #weights=None

            map_loader = ds.map_images_loader(
                preprocess=teacher_method.preprocess,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory)
            map_desc = teacher_method.compute_map_desc(dataloader=map_loader)
            del map_loader
        

            (h, w) = get_resize_size(student_method.preprocess)
            multipliers = np.linspace(args.min_mult, args.max_mult, args.n_points)
            resolutions = [(int(mult*h), int(mult*w)) for mult in multipliers]
            rec, res = [], []
            for idx, size in enumerate(resolutions):
                preprocess = replace_resize_transform(student_method.preprocess, size)
                query_loader = ds.query_images_loader(
                    preprocess=preprocess,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_memory,
                )
                _ = student_method.compute_query_desc(dataloader=query_loader)
                del query_loader
                student_method.set_map(map_desc)
                rec.append(recallatk(student_method, ds.ground_truth(), k=1))
                res.append(multipliers[idx])
            del student_method
            results[dataset_name][method_name + "_" + args.training_type[i]] = [res, rec]
        with open(join('Plots/PlaceRec/data/', dataset_name + '.pkl'), 'wb') as file:
            pickle.dump(results[dataset_name], file)
        del ds

############################### Plotting results ############################################

if args.mode == "plot":
    datasets = glob("Plots/PlaceRec/data/*.pkl")
    names = [pth.split('/')[-1].replace('.pkl', '') for pth in datasets]
    results = []
    for dataset in datasets:
        with open(dataset, 'rb') as file:
            results.append(pickle.load(file))

    for idx, dataset_results in enumerate(results):
        plot_curve(dataset_results, names[idx])
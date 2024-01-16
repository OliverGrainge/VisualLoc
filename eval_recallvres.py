import matplotlib.pyplot as plt
import os
from glob import glob
import pickle

def plot_resolution_curve(results: dict, dataset_name):
    fig, ax = plt.subplots()
    method_names = list(results.keys())
    for method_name in method_names:
        recalls = results[method_name][2]

        resolutions = results[method_name][0]
        plt.plot(resolutions, recalls, label=method_name)
    plt.title(dataset_name)
    plt.legend()
    plt.xlabel("Relative Image Scales")
    plt.ylabel("Recall@1")
    pth = os.getcwd() + "/Plots/PlaceRec/RecallvResolution"
    if not os.path.exists(pth):
        os.makedirs(pth)
    fig.savefig(pth + "/" + dataset_name + ".png")
    return ax


def plot_flops_curve(results: dict, dataset_name):
    fig, ax = plt.subplots()
    method_names = list(results.keys())
    for method_name in method_names:
        print(len(results[method_name]))
        recalls = results[method_name][2]
        flops = results[method_name][1]
        plt.plot(flops, recalls, label=method_name)
    plt.title(dataset_name)
    plt.legend()
    plt.xlabel("Flops Scales")
    plt.ylabel("Recall@1")
    pth = os.getcwd() + "/Plots/PlaceRec/RecallvFlops"
    if not os.path.exists(pth):
        os.makedirs(pth)
    fig.savefig(pth + "/" + dataset_name + ".png")
    return ax


datasets = glob("Data/*.pkl")
names = [pth.split('/')[-1].replace('.pkl', '') for pth in datasets]
results = []
for dataset in datasets:
    with open(dataset, 'rb') as file:
        results.append(pickle.load(file))



for idx, dataset_results in enumerate(results):
    plot_resolution_curve(dataset_results, names[idx])

for idx, dataset_results in enumerate(results):
    plot_flops_curve(dataset_results, names[idx])
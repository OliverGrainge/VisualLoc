
import argparse
import pandas as pd
import yaml
from PlaceRec.Training.dataloaders.val.MapillaryDataset import MSLS
from parsers import eval_arguments
import torchvision.transforms as T
import torch

args = eval_arguments()

from PlaceRec.Metrics import (average_precision, benchmark_latency_cpu,
                              benchmark_latency_gpu, count_flops, count_params,
                              measure_memory, plot_metric, plot_pr_curve,
                              recall_at_100p, recallatk)
from PlaceRec.utils import get_dataset, get_method
from PlaceRec.Quantization import quantize_model_trt
from PlaceRec.Training.dataloaders.val.CrossSeasonDataset import CrossSeasonDataset
from torch.utils.data import Subset


datasets = ["stlucia", "spedtest", "sfu", "pitts250k", "pitts30k", "nordland",
            "mapillarysls", "inriaholidays", "gardenspointwalking", "essex3in1",
            "crossseasons"]

accuracy_metrics = ["recall@1",
        "recall@5",
        "recall@10",
        "recall@100p",
        "average_precision"]

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}

valid_transform = T.Compose([
            T.Resize((320, 320), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"])])


dataset_accuracy_metrics = []

for ds in datasets:
    for am in accuracy_metrics:
        dataset_accuracy_metrics.append(ds + "_" + am)

for dataset_name in args.datasets:
    ds = get_dataset(dataset_name)
    ground_truth = ds.ground_truth()
    
    all_methods = []

    all_similarity = {}
    all_flops = {}
    all_params = {}
    recallat1 = {}
    recallat5 = {}
    recallat10 = {}
    recallat100p = {}
    latency_gpu_desktop = {}
    latency_gpu_embedded = {}
    latency_cpu_desktop = {}
    latency_cpu_embedded = {}
    memory = {}
    aupr = {}
   

    try:
        df = pd.read_csv("Plots/PlaceRec/data/results.csv")

        df.set_index("id", inplace=True)
    except:
        df = pd.DataFrame(
            columns=[
                "id",
                "dataset",
                "descriptor_bytes",
                "descriptor_dim",
                "flops",
                "params",
                "cpu_embedded_latency",
                "cpu_desktop_latency",
                "gpu_embedded_latency",
                "gpu_desktop_latency",
                "memory",
            ]  + dataset_accuracy_metrics
        )

        df.set_index("id", inplace=True)
    for method_name in args.methods:


        pretrained=False
        for m in args.metrics:
            if m in accuracy_metrics:
                pretrained=True
        

        if method_name == "quantvpr":
            method = get_method(method_name, pretrained=pretrained, args=args)
        else: 
            method = get_method(method_name, pretrained=pretrained)

        try:
            table_data = df.loc[method.name + "_" + args.precision]
        except:
            table_data = {}

        cal_ds = CrossSeasonDataset(valid_transform)
        cal_ds = Subset(cal_ds, list(range(100)))

        for m in args.metrics: 
            if "latency" in m: 
                method.model = quantize_model_trt(method.model, precision=args.precision, force_recalibration=True, model_name=method.name, descriptor_size=args.descriptor_size)

                #method.model = quantize_model(method.model, precision=args.precision, batch_size=1, calibration_dataset=cal_ds, calib_name=method.name + "_" + str(args.precision), reload_calib=True)
                break
        # only load pretrained weights if required

    
        # if measuring latency no need to load descriptors
        if pretrained:
            method.load_descriptors(ds.name)
        if isinstance(method.model, torch.nn.Module):
            method.set_device(args.device)
        
        if args.backend == "tensorrt":
            method.set_device("cuda")
            if args.precision == "fp32":
                method.model = quantize_model(method.model, precision="fp32", batch_size=1)
            elif args.precision == "fp16":
                method.model = quantize_model(method.model, precision="fp16", batch_size=1)
            elif args.precision == "int8":
                method.model = quantize_model(method.model, precision="int8", calibration_dataset=None, batch_size=1,
                                              reload_calib=args.reload_calib, calib_name=method.name + ".cache")

        all_methods.append(method)

        if pretrained:
            table_data["descriptor_bytes"] = (
                method.query_desc.nbytes / method.query_desc.shape[0]
            )
            table_data["descriptor_dim"] = method.query_desc.shape[1]

        if "count_flops" in args.metrics:
            all_flops[method.name] = count_flops(method)
            table_data["flops"] = all_flops[method.name]

        if "count_params" in args.metrics:
            all_params[method.name] = count_params(method)
            table_data["params"] = all_params[method.name]

        if "gpu_desktop_latency" in args.metrics:
            latency_gpu_desktop[method.name] = benchmark_latency_gpu(method)
            table_data["gpu_desktop_latency"] = latency_gpu_desktop[method.name]

        if "gpu_embedded_latency" in args.metrics:
            latency_gpu_embedded[method.name] = benchmark_latency_gpu(method)
            table_data["gpu_embedded_latency"] = latency_gpu_embedded[method.name]

        if "cpu_desktop_latency" in args.metrics:
            latency_cpu_desktop[method.name] = benchmark_latency_cpu(method)
            table_data["cpu_desktop_latency"] = latency_cpu_desktop[method.name]

        if "cpu_embedded_latency" in args.metrics:
            latency_cpu_embedded[method.name] = benchmark_latency_cpu(method)
            table_data["cpu_embedded_latency"] = latency_cpu_embedded[method.name]

        if "measure_memory" in args.metrics:
            memory[method.name] = measure_memory(args, method, jit=True)
            table_data["memory"] = memory[method.name]



        if "average_precision" in args.metrics:
            aupr[method.name] = average_precision(
                method=method, ground_truth=ground_truth
            )
            table_data[ds.name + "_average_precision"] = aupr[method.name]

        if "recall@1" in args.metrics:
            recallat1[method.name] = recallatk(
                method=method,
                ground_truth=ground_truth,
                k=1,
            )
            table_data[ds.name + "_recall@1"] = recallat1[method.name]

        if "recall@100p" in args.metrics:
            recallat100p[method.name] = recall_at_100p(
                method=method,
                ground_truth=ground_truth,
            )
            table_data[ds.name + "_recall@100p"] = recallat100p[method.name]

        if "recall@5" in args.metrics:
            recallat5[method.name] = recallatk(
                method=method,
                ground_truth=ground_truth,
                k=5,
            )

            table_data[ds.name + "_recall@5"] = recallat5[method.name]

        if "recall@10" in args.metrics:
            recallat10[method.name] = recallatk(
                method=method,
                ground_truth=ground_truth,
                k=10,
            )
            table_data[ds.name + "_recall@10"] = recallat10[method.name]
            table_data["dataset"] = dataset_name

        df.loc[method.name + "_" + args.precision] = table_data

    if "prcurve" in args.metrics:
        plot_pr_curve(
            methods=all_methods,
            ground_truth=ground_truth,
            show=False,
            title="PR Curve for " + ds.name,
            dataset_name=ds.name,
        )

    if "measure_memory" in args.metrics:
        plot_metric(
            methods=list(memory.keys()),
            scores=list(memory.values()),
            show=False,
            metric_name="memory_consumption",
            title="Disk Memory Consumption (Mb)",
            dataset_name=ds.name,
        )

    if "average_precision" in args.metrics:
        plot_metric(
            methods=list(aupr.keys()),
            scores=list(aupr.values()),
            show=False,
            metric_name="average_precision",
            title="Average Precision",
            dataset_name=ds.name,
        )

    if "recall@100p" in args.metrics:
        plot_metric(
            methods=list(recallat100p.keys()),
            title="recall@100precision",
            show=False,
            metric_name="recall@100p",
            dataset_name=ds.name,
            scores=list(recallat100p.values()),
        )

    if "gpu_latency_desktop" in args.metrics:
        plot_metric(
            methods=list(latency_gpu_desktop.keys()),
            scores=list(latency_gpu_desktop.values()),
            title="Desktop GPU Latency (ms)",
            show=False,
            metric_name="gpu_latency_desktop",
            dataset_name=ds.name,
        )

    if "cpu_latency_desktop" in args.metrics:
        plot_metric(
            methods=list(latency_cpu_desktop.keys()),
            scores=list(latency_cpu_desktop.values()),
            title="Desktop CPU Latency (ms)",
            show=False,
            metric_name="cpu_latency_desktop",
            dataset_name=ds.name,
        )

    if "cpu_latency_embedded" in args.metrics:
        plot_metric(
            methods=list(latency_cpu_embedded.keys()),
            scores=list(latency_cpu_embedded.values()),
            title="CPU Embedded Latency (ms)",
            show=False,
            metric_name="cpu_latency_embedded",
            dataset_name=ds.name,
        )

    if "gpu_latency_embedded" in args.metrics:
        plot_metric(
            methods=list(latency_gpu_embedded.keys()),
            scores=list(latency_gpu_embedded.values()),
            title="GPU Embedded Latency (ms)",
            show=False,
            metric_name="gpu_latency_embedded",
            dataset_name=ds.name,
        )

    if "count_flops" in args.metrics:
        plot_metric(
            methods=list(all_flops.keys()),
            scores=list(all_flops.values()),
            title="FLOP Count for Models",
            show=False,
            metric_name="count_flops",
            dataset_name=ds.name,
        )

    if "count_params" in args.metrics:
        plot_metric(
            methods=list(all_flops.keys()),
            scores=list(all_flops.values()),
            title="Param Count for Models",
            show=False,
            metric_name="count_param",
            dataset_name=ds.name,
        )

    if "recall@1" in args.metrics:
        plot_metric(
            methods=list(recallat1.keys()),
            scores=list(recallat1.values()),
            title="recall@1 for: " + ds.name,
            show=False,
            metric_name="recall@1",
            dataset_name=ds.name,
        )

    if "recall@5" in args.metrics:
        plot_metric(
            methods=list(recallat5.keys()),
            scores=list(recallat5.values()),
            title="recall@5 for: " + ds.name,
            show=False,
            metric_name="recall@5",
            dataset_name=ds.name,
        )

    if "recall@10" in args.metrics:
        plot_metric(
            methods=list(recallat10.keys()),
            scores=list(recallat10.values()),
            title="recall@10 for: " + ds.name,
            show=False,
            metric_name="recall@10",
            dataset_name=ds.name,
        )

df.to_csv("Plots/PlaceRec/data/results.csv")

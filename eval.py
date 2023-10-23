import argparse

import pandas as pd

from PlaceRec.Metrics import (
    average_precision,
    benchmark_latency_cpu,
    benchmark_latency_gpu,
    count_flops,
    count_params,
    plot_dataset_sample,
    plot_metric,
    plot_pr_curve,
    recall_at_100p,
    recallatk,
)
from PlaceRec.utils import get_dataset, get_method

parser = argparse.ArgumentParser()

parser.add_argument(
    "--datasets",
    choices=(
        "gsvcities",
        "sfu",
        "gardenspointwalking",
        "stlucia_small",
        "essex3in1",
        "nordlands",
    ),
    help="specify one of the datasets from PlaceRec.Datasets",
    type=str,
    default=["stlucia_small"],
    nargs="+",
)

parser.add_argument(
    "--methods",
    choices=(
        "regionvlad",
        "mixvpr",
        "convap",
        "amosnet",
        "hybridnet",
        "netvlad",
        "hog",
        "cosplace",
        "calc",
        "alexnet",
    ),
    help="specify one of the techniques from vpr/vpr_tecniques",
    type=str,
    default="hog",
    nargs="+",
)

parser.add_argument(
    "--partition",
    type=str,
    default="test",
    help="choose from 'train', 'val', 'test' or 'all'",
)

parser.add_argument("--metrics", type=str, default="prcurve", nargs="+")


args = parser.parse_args()

for dataset_name in args.datasets:
    ds = get_dataset(dataset_name)
    gt_hard = ds.ground_truth(partition=args.partition, gt_type="hard")
    gt_soft = ds.ground_truth(partition=args.partition, gt_type="soft")

    all_similarity = {}
    all_flops = {}
    all_params = {}
    recallat1 = {}
    recallat5 = {}
    recallat10 = {}
    recallat100p = {}
    latency_cpu = {}
    latency_gpu = {}
    aupr = {}

    try:
        df = pd.read_csv("Plots/results.csv")
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
                "gpu_latency",
                "cpu_latency",
                "recall@1",
                "recall@5",
                "recall@10",
                "recall@100p",
                "average_precision",
            ]
        )

        df.set_index("id", inplace=True)

    for method_name in args.methods:
        table_data = {}
        results_id = method_name + "_" + dataset_name
        method = get_method(method_name)
        method.load_descriptors(ds.name)
        similarity = method.similarity_matrix(method.query_desc, method.map_desc)
        all_similarity[method.name] = similarity
        ground_truth = ds.ground_truth(partition=args.partition, gt_type="hard")
        ground_truth_soft = ds.ground_truth(partition=args.partition, gt_type="soft")
        table_data["descriptor_bytes"] = method.query_desc["query_descriptors"].nbytes / method.query_desc["query_descriptors"].shape[0]
        table_data["descriptor_dim"] = method.query_desc["query_descriptors"].shape[1]

        if "count_flops" in args.metrics:
            all_flops[method.name] = count_flops(method)
            table_data["flops"] = all_flops[method.name]

        if "count_params" in args.metrics:
            all_params[method.name] = count_params(method)
            table_data["params"] = all_params[method.name]

        if "gpu_latency" in args.metrics:
            latency_gpu[method.name] = benchmark_latency_gpu(method)
            table_data["gpu_latency"] = latency_gpu[method.name]

        if "cpu_latency" in args.metrics:
            latency_cpu[method.name] = benchmark_latency_cpu(method)
            table_data["cpu_latency"] = latency_cpu[method.name]

        if "average_precision" in args.metrics:
            aupr[method.name] = average_precision(ground_truth=ground_truth, ground_truth_soft=ground_truth_soft, similarity=similarity)
            table_data["average_precision"] = aupr[method.name]

        if "recall@1" in args.metrics:
            recallat1[method.name] = recallatk(
                ground_truth=ground_truth,
                ground_truth_soft=ground_truth_soft,
                similarity=similarity,
                k=1,
            )
            table_data["recall@1"] = recallat1[method.name]

        if "recall@100p" in args.metrics:
            recallat100p[method.name] = recall_at_100p(
                ground_truth=ground_truth,
                ground_truth_soft=ground_truth_soft,
                similarity=similarity,
                k=1,
                matching="single",
            )
            table_data["recall@100p"] = recallat100p[method.name]

        if "recall@5" in args.metrics:
            recallat5[method.name] = recallatk(
                ground_truth=ground_truth,
                ground_truth_soft=ground_truth_soft,
                similarity=similarity,
                k=5,
            )

            table_data["recall@5"] = recallat5[method.name]

        if "recall@10" in args.metrics:
            recallat10[method.name] = recallatk(
                ground_truth=ground_truth,
                ground_truth_soft=ground_truth_soft,
                similarity=similarity,
                k=10,
            )
            table_data["recall@10"] = recallat10[method.name]
            table_data["dataset"] = dataset_name

        df.loc[results_id] = table_data

    if "prcurve" in args.metrics:
        plot_pr_curve(
            ground_truth=gt_hard,
            all_similarity=all_similarity,
            ground_truth_soft=gt_soft,
            matching="single",
            show=False,
            title="PR Curve for " + ds.name + " partition: " + args.partition,
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
        )

    if "gpu_latency" in args.metrics:
        plot_metric(
            methods=list(latency_gpu.keys()),
            scores=list(latency_gpu.values()),
            title="GPU Latency (ms)",
            show=False,
            metric_name="gpu_latency",
            dataset_name=ds.name,
        )

    if "cpu_latency" in args.metrics:
        plot_metric(
            methods=list(latency_cpu.keys()),
            scores=list(latency_cpu.values()),
            title="CPU Latency (ms)",
            show=False,
            metric_name="cpu_latency",
            dataset_name=ds.name,
        )

    if "dataset_sample" in args.metrics:
        plot_dataset_sample(ds, gt_type="soft", show=False)

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

df.to_csv("Plots/results.csv")

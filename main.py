import argparse

from PlaceRec.Metrics import (
    count_flops,
    count_params,
    plot_dataset_sample,
    plot_metric,
    plot_pr_curve,
    recallatk,
)
from PlaceRec.utils import get_dataset, get_method

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    required=True,
    choices=("describe", "evaluate"),
    help="Specify either describe or evaluate",
    type=str,
)
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
parser.add_argument("--batchsize", type=int, default=10, help="Choose the Batchsize for VPR processing")
parser.add_argument(
    "--partition",
    type=str,
    default="test",
    help="choose from 'train', 'val', 'test' or 'all'",
)
parser.add_argument("--metrics", type=str, default="prcurve", nargs="+")
parser.add_argument(
    "--num_workers",
    type=int,
    default=16,
    help="Choose the number of processing the threads for the dataloader",
)
args = parser.parse_args()


if args.mode == "describe":
    for method_name in args.methods:
        method = get_method(method_name)
        for dataset_name in args.datasets:
            ds = get_dataset(dataset_name)
            map_loader = ds.map_images_loader(
                partition=args.partition,
                preprocess=method.preprocess,
                num_workers=args.num_workers,
            )
            _ = method.compute_map_desc(dataloader=map_loader)
            del map_loader
            query_loader = ds.query_images_loader(
                partition=args.partition,
                preprocess=method.preprocess,
                num_workers=args.num_workers,
            )
            _ = method.compute_query_desc(dataloader=query_loader)
            del query_loader
            method.save_descriptors(ds.name)
            del ds
        del method

elif args.mode == "evaluate":
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
        for method_name in args.methods:
            method = get_method(method_name)
            method.load_descriptors(ds.name)
            similarity = method.similarity_matrix(method.query_desc, method.map_desc)
            all_similarity[method.name] = similarity
            ground_truth = ds.ground_truth(partition=args.partition, gt_type="hard")
            ground_truth_soft = ds.ground_truth(partition=args.partition, gt_type="soft")

            if "count_flops" in args.metrics:
                all_flops[method.name] = count_flops(method)

            if "count_params" in args.metrics:
                print(count_params(method))
                all_params[method.name] = count_params(method)

            if "recall@1":
                recallat1[method.name] = recallatk(
                    ground_truth=ground_truth,
                    ground_truth_soft=ground_truth_soft,
                    similarity=similarity,
                    k=1,
                )

            if "recall@5":
                recallat5[method.name] = recallatk(
                    ground_truth=ground_truth,
                    ground_truth_soft=ground_truth_soft,
                    similarity=similarity,
                    k=5,
                )

            if "recall@10":
                recallat10[method.name] = recallatk(
                    ground_truth=ground_truth,
                    ground_truth_soft=ground_truth_soft,
                    similarity=similarity,
                    k=10,
                )

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

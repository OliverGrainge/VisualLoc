import argparse
from PlaceRec.Metrics import (
    plot_pr_curve,
    plot_dataset_sample,
    count_flops,
    count_params,
    plot_metric,
    recallatk,
    plot_selection_histogram,
)
from PlaceRec.utils import get_dataset, get_method

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    required=True,
    choices=("describe", "evaluate"),
    help="Specify either describe or evaluate or both",
    type=str,
    nargs="+",
)
parser.add_argument(
    "--datasets",
    help="specify one of the datasets from PlaceRec.Datasets",
    type=str,
    default=["stlucia_small"],
    nargs="+",
)
parser.add_argument(
    "--methods",
    choices=(
        "multiplexvpr",
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
        "densevlad",
    ),
    help="specify one of the techniques from vpr/vpr_tecniques",
    type=str,
    default="hog",
    nargs="+",
)
parser.add_argument(
    "--batchsize", type=int, default=10, help="Choose the Batchsize for VPR processing"
)
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


if "describe" in args.mode:
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
            # del map_loader
            query_loader = ds.query_images_loader(
                partition=args.partition,
                preprocess=method.preprocess,
                num_workers=args.num_workers,
            )
            if method_name == "multiplexvpr":
                _ = method.compute_query_desc(
                    query_dataloader=query_loader, map_dataloader=map_loader
                )
            else:
                _ = method.compute_query_desc(dataloader=query_loader)
            del query_loader
            method.save_descriptors(ds.name)
            del ds
        del method

elif "evaluate" in args.mode:
    for dataset_name in args.datasets:
        ds = get_dataset(dataset_name)
        ground_truth = ds.ground_truth(partition=args.partition, gt_type="hard")
        ground_truth_soft = ds.ground_truth(partition=args.partition, gt_type="soft")

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
            if "multiplexvpr_selections" and method_name == "multiplexvpr":
                plot_selection_histogram(
                    query_desc=method.query_desc,
                    methods=method.methods,
                    dataset_name=dataset_name,
                )

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
                ground_truth=ground_truth,
                all_similarity=all_similarity,
                ground_truth_soft=ground_truth_soft,
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

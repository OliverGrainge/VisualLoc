import argparse
from PlaceRec.Metrics import plot_pr_curve
from PlaceRec.utils import get_dataset, get_method

parser = argparse.ArgumentParser()

parser.add_argument('--mode', required=True, choices=("describe", "evaluate"),
                    help='Specify either describe or evaluate', type=str)
parser.add_argument('--datasets', choices=("sfu", "gardenspointwalking", "stlucia_small"),
                    help='specify one of the datasets from PlaceRec.Datasets', type=str, default=["stlucia_small"], nargs='+')
parser.add_argument('--methods', choices=("netvlad", "hog", "cosplace", "calc", "alexnet"),
                    help="specify one of the techniques from vpr/vpr_tecniques", type=str, default="hog", nargs='+')
parser.add_argument('--batchsize', type=int, default=10, help="Choose the Batchsize for VPR processing")
parser.add_argument('--partition', type=str, default='test', help="choose from 'train', 'val', 'test' or 'all'")
parser.add_argument('--metrics', type=str, default='prcurve', nargs='+')
args = parser.parse_args()






if args.mode == "describe":
    for method_name in args.methods:
        method = get_method(method_name)
        for dataset_name in args.datasets:
            ds = get_dataset(dataset_name)
            query_loader = ds.query_images_loader(partition=args.partition, preprocess=method.preprocess)
            _ = method.compute_map_desc(dataloader=query_loader)
            del query_loader 
            map_loader = ds.map_images_loader(preprocess=method.preprocess)
            _ = method.compute_query_desc(dataloader=map_loader)
            del map_loader 
            method.save_descriptors(ds.name)
            del ds 
        del method

elif args.mode == "evaluate":
    for dataset_name in args.datasets:
        ds = get_dataset(dataset_name)
        gt_hard = ds.ground_truth(partition=args.partition, gt_type="hard")
        gt_soft = ds.ground_truth(partition=args.partition, gt_type="soft")

        all_similarity = {}
        for method_name in args.methods:
            method = get_method(method_name)
            method.load_descriptors(ds.name)
            similarity = method.similarity_matrix(method.query_desc, method.map_desc)
            all_similarity[method.name] = similarity

        if "prcurve" in args.metrics:
            plot_pr_curve(ground_truth=gt_hard, 
                            all_similarity=all_similarity, 
                            ground_truth_soft=gt_soft,
                            matching="single",
                            show=False, 
                            title="PR Curve for " + ds.name + " partition: " + args.partition, 
                            dataset_name=ds.name)
        
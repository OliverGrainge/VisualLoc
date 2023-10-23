import argparse

from PlaceRec.utils import get_dataset, get_method

parser = argparse.ArgumentParser()

######################################### Arguments ######################################

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

parser.add_argument(
    "--num_workers",
    type=int,
    default=0,
    help="Choose the number of processing the threads for the dataloader",
)

parser.add_argument("--pin_memory", type=bool, default=False, help="Choose whether to pin memory in GPU")

args = parser.parse_args()


#################################### Computing Descriptors ###########################################


for method_name in args.methods:
    method = get_method(method_name)
    for dataset_name in args.datasets:
        ds = get_dataset(dataset_name)
        map_loader = ds.map_images_loader(
            partition=args.partition, preprocess=method.preprocess, num_workers=args.num_workers, pin_memory=args.pin_memory
        )
        _ = method.compute_map_desc(dataloader=map_loader)
        del map_loader
        query_loader = ds.query_images_loader(
            partition=args.partition,
            preprocess=method.preprocess,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        _ = method.compute_query_desc(dataloader=query_loader)
        del query_loader
        method.save_descriptors(ds.name)
        del ds
    del method

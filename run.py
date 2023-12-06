import argparse

import yaml

from parsers import run_arguments
from PlaceRec.utils import get_config, get_dataset, get_method

config = get_config()
args = run_arguments()


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

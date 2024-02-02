import argparse

import yaml

#from parsers import run_arguments
from PlaceRec.utils import get_config, get_dataset, get_method
from parsers import run_arguments
from PlaceRec.Quantization import quantize_model
from PlaceRec.Training.dataloaders.val.MapillaryDataset import MSLS 
from torch.utils.data import Subset
config = get_config()
args = run_arguments()

print(args)

#################################### Computing Descriptors ###########################################
for method_name in args.methods:
    if method_name == "quantvpr":
        method = get_method(method_name, pretrained=True, args=args)
    else: 
        method = get_method(method_name, pretrained=True)
    method.set_device(args.device)
    cal_ds = MSLS(method.preprocess)
    cal_ds = Subset(cal_ds, list(range(100)))
    method.model = quantize_model(method.model, precision=args.precision, calibration_dataset=cal_ds, batch_size=1, calib_name=method.name + "_" + str(args.precision), reload_calib=False)
    for dataset_name in args.datasets:
        ds = get_dataset(dataset_name)
        map_loader = ds.map_images_loader(
            preprocess=method.preprocess,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            batch_size=1
        )
        _ = method.compute_map_desc(dataloader=map_loader)
        del map_loader
        query_loader = ds.query_images_loader(
            preprocess=method.preprocess,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            batch_size=1
        )
        _ = method.compute_query_desc(dataloader=query_loader)
        del query_loader
        method.save_descriptors(ds.name)
        del ds
    del method


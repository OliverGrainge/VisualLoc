import argparse

import yaml

#from parsers import run_arguments
from PlaceRec.utils import get_config, get_dataset, get_method
from parsers import run_arguments
from PlaceRec.Quantization import quantize_model_trt
from PlaceRec.Training.dataloaders.val.MapillaryDataset import MSLS 
from torch.utils.data import Subset
import torch
config = get_config()
args = run_arguments()

#################################### Computing Descriptors ###########################################
for method_name in args.methods:
    if method_name == "quantvpr":
        method = get_method(method_name, pretrained=True, args=args)
    else: 
        method = get_method(method_name, pretrained=True)

    method.set_device(args.device)
    #method.model = quantize_model(method.model, precision=args.precision, calibration_dataset=cal_ds, batch_size=32, calib_name=method.name + "_" + str(args.precision), reload_calib=False)
    #print("======================", method.name)
    method.model = quantize_model_trt(method.model, precision=args.precision, force_recalibration=True, descriptor_size=method.features_dim)
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


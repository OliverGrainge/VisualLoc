import argparse
import os
from os.path import join

import torch
import yaml

from PlaceRec.utils import get_config

config = get_config()

################################## RUN Mode Argument Parser ##################################


def run_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        help="specify one of the datasets from PlaceRec.Datasets",
        type=str,
        default=config["run"]["datasets"],
        nargs="+",
    )

    parser.add_argument(
        "--methods",
        help="specify one of the techniques from vpr/vpr_tecniques",
        type=str,
        default=config["run"]["methods"],
        nargs="+",
    )

    parser.add_argument(
        "--batchsize",
        type=int,
        default=config["run"]["batchsize"],
        help="Choose the Batchsize for VPR processing",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=config["run"]["num_workers"],
        help="Choose the number of processing the threads for the dataloader",
    )

    parser.add_argument(
        "--pin_memory",
        type=bool,
        default=config["run"]["pin_memory"],
        help="Choose whether to pin memory in GPU",
    )

    parser.add_argument("--device", type=str, default=config["run"]["device"])

    args = parser.parse_args()
    return args


################################## EVAL Mode Argument Parser ##################################


def eval_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        help="specify one of the datasets from PlaceRec.Datasets",
        type=str,
        default=config["eval"]["datasets"],
        nargs="+",
    )

    parser.add_argument(
        "--methods",
        help="specify one of the techniques from vpr/vpr_tecniques",
        type=str,
        default=config["eval"]["methods"],
        nargs="+",
    )

    parser.add_argument(
        "--partition",
        type=str,
        default=config["eval"]["partition"],
        help="choose from 'train', 'val', 'test' or 'all'",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=config["eval"]["device"],
        help="choose the device to benchmark inference",
    )

    parser.add_argument(
        "--metrics", type=str, default=config["eval"]["metrics"], nargs="+"
    )

    args = parser.parse_args()
    return args


def train_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--training_method", type=str, default=config["train"]["training_method"]
    )

    parser.add_argument("--method", type=str, default=config["train"]["method"])

    parser.add_argument(
        "--pretrained", type=bool, default=config["train"]["pretrained"]
    )

    parser.add_argument(
        "--device", type=str, default=config["train"]["device"]
    )

    parser.add_argument(
        "--image_resolution",
        type=int,
        default=config["train"]["image_resolution"],
        nargs="+",
    )

    parser.add_argument("--batch_size", type=int, default=config["train"]["batch_size"])

    args = parser.parse_args()
    return args

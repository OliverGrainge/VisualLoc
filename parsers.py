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

    parser.add_argument(
        "--device",
        type=str,
        default=config["run"]["device"]
    )

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


def train_args(): 
    parser = argparse.ArgumentParser()

    config = get_config()

    parser.add_argument(
        "--datasets_directory",
        type=str,
        default=config["datasets_directory"]
    )

    parser.add_argument(
        "--weights_directory",
        type=str,
        default=config["weights_directory"]
    )

    config = config["train"]

    parser.add_argument(
        "--student_method",
        type=str,
        default=config["student_method"]
    )


    parser.add_argument(
        "--teacher_method",
        type=str,
        default=config["teacher_method"]
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=config["batch_size"]
    )

    parser.add_argument(
        "--device",
        type=str,
        default=config["device"]
    )

    parser.add_argument(
        "--reload",
        type=bool,
        default=config["reload"]
    )

    parser.add_argument(
        "--distillation_type",
        type=str,
        default=config["distillation_type"]
    )

    parser.add_argument(
        "--min_mult",
        type=float,
        default=config["min_mult"]
    )


    parser.add_argument(
        "--max_mult",
        type=float,
        default=config["max_mult"]
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=config["patience"]
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=config["learning_rate"]
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=config["num_workers"]
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=config["max_epochs"]
    )

    parser.add_argument(
        "--size",
        type=int,
        default=config["size"],
        nargs="+"
    )

    args = parser.parse_args()
    return args


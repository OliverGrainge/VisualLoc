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
        "--backbone", 
        type=str,
        default=config["train"]["backbone"]
    )

    parser.add_argument(
        "--aggregation", 
        type=str,
        default=config["train"]["aggregation"]
    )


    parser.add_argument(
        "--optimizer", 
        type=str,
        default=config["train"]["optimizer"]
    )


    parser.add_argument(
        "--loss_name", 
        type=str,
        default=config["train"]["loss_name"]
    )

    parser.add_argument(
        "--miner_name", 
        type=str,
        default=config["train"]["miner_name"]
    )

    parser.add_argument(
        "--miner_margin", 
        type=float,
        default=config["train"]["miner_margin"]
    )
    
    parser.add_argument(
        "--lr", 
        type=float,
        default=config["train"]["lr"]
    )

    parser.add_argument(
        "--weight_decay", 
        type=float,
        default=config["train"]["weight_decay"]
    )

    parser.add_argument(
        "--momentum", 
        type=float,
        default=config["train"]["momentum"]
    )

    parser.add_argument(
        "--warmup_steps", 
        type=int,
        default=config["train"]["warmup_steps"]
    )

    parser.add_argument(
        "--milestones",
        type=int,
        default=config["train"]["milestones"],
        nargs="+"
    )

    parser.add_argument(
        "--lr_mult", 
        type=float,
        default=config["train"]["lr_mult"]
    )

    parser.add_argument(
        "--descriptor_size",
        type=int,
        default=config["train"]["descriptor_size"]
    )
    args = parser.parse_args()
    return args


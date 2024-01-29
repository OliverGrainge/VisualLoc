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
        "--datasets_directory",
        type=str,
        default=config["datasets_directory"]
    )


    parser.add_argument(
        "--device",
        type=str,
        default=config["train"]["device"],
        help="Choose the number of processing the threads for the dataloader",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=config["train"]["batch_size"],
        help="Choose the number of processing the threads for the dataloader",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=config["train"]["learning_rate"],
        help="Choose the number of processing the threads for the dataloader",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=config["train"]["num_workers"],
        help="Choose the number of processing the threads for the dataloader",
    )

    parser.add_argument("--seed", type=int, default=config["train"]["seed"], help="seed the training run")

    parser.add_argument(
        "--test_method",
        type=str,
        default="hard_resize",
        choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"],
        help="This includes pre/post-processing methods and prediction refinement",
    )

    parser.add_argument("--max_epochs", type=int, default=config["train"]["max_epochs"])


    parser.add_argument("--teacher_methods", type=str, default=config["train"]["teacher_methods"], nargs="+")

    parser.add_argument("--student_method", type=str, default=config["train"]["student_method"])

    parser.add_argument("--optimizer", type=str, default=config["train"]["optimizer"])

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=config["train"]["weight_decay"],
    )

    parser.add_argument("--momentum", type=float, default=config["train"]["momentum"])

    parser.add_argument("--fusion_method", type=str, default=config["train"]["fusion_method"])

    parser.add_argument("--warmup_steps", type=int, default=config["train"]["warmup_steps"])

    parser.add_argument("--faiss_gpu", type=bool, default=config["train"]["faiss_gpu"])

    parser.add_argument("--random_sample_from_each_place", type=bool, default=config["train"]["random_sample_from_each_place"])

    parser.add_argument("--image_size", type=int, nargs="+", default=config["train"]["image_size"])

    parser.add_argument("--show_data_stats", type=bool, default=config["train"]["show_data_stats"])

    parser.add_argument("--val_set_names", type=str, nargs="+", default=config["train"]["val_set_names"])

    parser.add_argument("--rkd_loss", type=str, default=config["train"]["rkd_loss"])\
    
    parser.add_argument("--logger_type", type=str, default=config["train"]["logger_type"])

    parser.add_argument("--lr_milestones", type=float, default=config["train"]["lr_milestones"], nargs="+")

    parser.add_argument("--lr_mult", type=float, default=config["train"]["lr_mult"])

    args = parser.parse_args()
    return args
import argparse
import os
from os.path import join

import torch
import yaml

# importing the config file. If any arguments are not specified in the command line
# they will be set to the values in the config file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

################################## RUN Mode Argument Parser ##################################


def run_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        choices=("gsvcities", "sfu", "gardenspointwalking", "stlucia_small", "essex3in1", "nordlands", "pitts30k", "stlucia_large", "msls"),
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

    parser.add_argument("--batchsize", type=int, default=config["run"]["batch_size"], help="Choose the Batchsize for VPR processing")

    parser.add_argument(
        "--partition",
        type=str,
        default=config["run"]["partition"],
        help="choose from 'train', 'val', 'test' or 'all'",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=config["run"]["num_workers"],
        help="Choose the number of processing the threads for the dataloader",
    )

    parser.add_argument("--pin_memory", type=bool, default=config["run"]["pin_memory"], help="Choose whether to pin memory in GPU")

    args = parser.parse_args()
    return args


################################## EVAL Mode Argument Parser ##################################


def eval_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        choices=("gsvcities", "sfu", "gardenspointwalking", "stlucia_small", "essex3in1", "nordlands", "pitts30k", "stlucia_large", "msls"),
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
        "--input_size",
        type=int,
        default=config["eval"]["input_size"],
        nargs=2,
        help="Resizing shape for images (HxW).",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=config["eval"]["device"],
        help="choose the device to benchmark inference",
    )

    parser.add_argument("--metrics", type=str, default="prcurve", nargs="+")

    args = parser.parse_args()
    return args


################################## TRAIN Mode Argument Parser ##################################


def train_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--min_size", type=int, default=config["train"]["min_size"])

    parser.add_argument("--max_size", type=int, default=config["train"]["max_size"])

    parser.add_argument("--recall_values", type=int, default=config["train"]["recall_values"], nargs="+", help="Recalls to be computed, such as R@5.")

    parser.add_argument("--distillation_type", type=str, default=config["train"]["distillation_type"])

    parser.add_argument(
        "--dataset_name",
        type=str,
        default=config["train"]["dataset_name"],
        help="Choose the number of processing the threads for the dataloader",
    )

    parser.add_argument(
        "--datasets_folder",
        type=str,
        default=join(config["train"]["datasets_folder"], "datasets_vg"),
        help="Choose the number of processing the threads for the dataloader",
    )


    parser.add_argument(
        "--val_positive_dist_threshold",
        type=int,
        default=config["train"]["val_positive_dist_threshold"],
        help="Choose the number of processing the threads for the dataloader",
    )

    parser.add_argument(
        "--train_positive_dist_threshold",
        type=int,
        default=config["train"]["train_positive_dist_threshold"],
        help="Choose the number of processing the threads for the dataloader",
    )

    parser.add_argument(
        "--mining",
        type=str,
        default=config["train"]["mining"],
        help="Choose the number of processing the threads for the dataloader",
    )

    parser.add_argument(
        "--infer_batch_size",
        type=int,
        default=config["train"]["infer_batch_size"],
        help="Choose the number of processing the threads for the dataloader",
    )

    parser.add_argument(
        "--cache_refresh_rate",
        type=int,
        default=config["train"]["cache_refresh_rate"],
        help="Choose the number of processing the threads for the dataloader",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=config["train"]["device"],
        help="Choose the number of processing the threads for the dataloader",
    )

    parser.add_argument(
        "--neg_num_per_query",
        type=int,
        default=config["train"]["neg_num_per_query"],
        help="Choose the number of processing the threads for the dataloader",
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=config["train"]["margin"],
        help="Choose the number of processing the threads for the dataloader",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=config["train"]["train_batch_size"],
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

    parser.add_argument("--patience", type=int, default=config["train"]["patience"], help="Choose the early stopping patience number")

    parser.add_argument("--seed", type=int, default=config["train"]["seed"], help="seed the training run")

    parser.add_argument(
        "--training_type",
        type=str,
        default=config["train"]["training_type"],
        help="seed the training run",
    )

    parser.add_argument(
        "--method",
        type=str,
        default=config["train"]["method"],
        help="seed the training run",
    )

    parser.add_argument(
        "--test_method",
        type=str,
        default="hard_resize",
        choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"],
        help="This includes pre/post-processing methods and prediction refinement",
    )

    parser.add_argument(
        "--max_epochs", 
        type=int, 
        default=config["train"]["max_epochs"])

    parser.add_argument(
        "--loss_distance", 
        type=str, 
        default=config["train"]["loss_distance"])

    parser.add_argument(
        "--val_check_interval", 
        type=int, 
        default=config["train"]["val_check_interval"])

    parser.add_argument(
        "--teacher_method", 
        type=str, 
        default=config["train"]["teacher_method"])

    parser.add_argument(
        "--student_method", 
        type=str, 
        default=config["train"]["teacher_method"])

    parser.add_argument(
        "--reload", 
        type=bool, 
        default=config["train"]["reload"])
    
    parser.add_argument(
        "--layers_to_freeze", 
        type=int,
        default=config["train"]["layers_to_freeze"])
    
    parser.add_argument(
        "--layers_to_crop", 
        type=str,
        nargs="+",
        default=config["train"]["layers_to_crop"]
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default=config["train"]["optimizer"]
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=config["train"]["weight_decay"],
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
        "--faiss_gpu",
        type=bool,
        default=config["train"]["faiss_gpu"]
    )

    parser.add_argument(
        "--pretrained",
        type=bool,
        default=config["train"]["pretrained"]
    )

    parser.add_argument(
        "--img_per_place",
        type=int,
        default=config["train"]["img_per_place"]
    )

    parser.add_argument(
        "--min_img_per_place",
        type=int,
        default=config["train"]["min_img_per_place"]
    )

    parser.add_argument(
        "--cities",
        type=str,
        nargs="+",
        default=config["train"]["cities"]
    )

    parser.add_argument(
        "--shuffle_all",
        type=bool,
        default=config["train"]["shuffle_all"]
    )

    parser.add_argument(
        "--random_sample_from_each_place",
        type=bool,
        default=config["train"]["random_sample_from_each_place"]
    )

    parser.add_argument(
        "--image_size",
        type=int,
        nargs="+",
        default=config["train"]["image_size"]
    )

    parser.add_argument(
        "--show_data_stats",
        type=bool,
        default=config["train"]["show_data_stats"]
    )

    parser.add_argument(
        "--val_set_names",
        type=str,
        nargs="+",
        default=config["train"]["val_set_names"]
    )

    parser.add_argument(
        "--batch_sampler",
        type=bool,
        default=config["train"]["batch_sampler"]
    )
    

    

    args = parser.parse_args()
    return args

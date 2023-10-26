import argparse
import yaml

with open('config.yaml', 'r') as file:
    args = yaml.safe_load(file)

train_args = args["train"]

parser = argparse.ArgumentParser(
    description="Training Visual Geolocalization",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
# Training parameters
parser.add_argument(
    "--train_batch_size",
    type=int,
    default=train_args["train_batch_size"],
    help="Number of triplets (query, pos, negs) in a batch. Each triplet consists of 12 images",
)




parser.add_argument(
    "--infer_batch_size",
    type=int,
    default=train_args["infer_batch_size"],
    help="Batch size for inference (caching and testing)",
)
parser.add_argument(
    "--criterion",
    type=str,
    default=train_args["criterion"],
    help="loss to be used",
    choices=["triplet", "sare_ind", "sare_joint"],
)
parser.add_argument(
    "--margin", type=float, default=train_args["margin"], help="margin for the triplet loss"
)
parser.add_argument(
    "--epochs_num", type=int, default=train_args["epochs_num"], help="number of epochs to train for"
)
parser.add_argument("--patience", type=int, default=train_args["patience"])
parser.add_argument("--lr", type=float, default=train_args["lr"], help="_")

parser.add_argument(
    "--optim", type=str, default=train_args["optim"], help="_", choices=["adam", "sgd"]
)
parser.add_argument(
    "--cache_refresh_rate",
    type=int,
    default=train_args["cache_refresh_rate"],
    help="How often to refresh cache, in number of queries",
)
parser.add_argument(
    "--queries_per_epoch",
    type=int,
    default=train_args["queries_per_epoch"],
    help="How many queries to consider for one epoch. Must be multiple of cache_refresh_rate",
)
parser.add_argument(
    "--negs_num_per_query",
    type=int,
    default=train_args["negs_num_per_query"],
    help="How many negatives to consider per each query in the loss",
)
parser.add_argument(
    "--neg_samples_num",
    type=int,
    default=train_args["neg_samples_num"],
    help="How many negatives to use to compute the hardest ones",
)
parser.add_argument(
    "--mining",
    type=str,
    default=train_args["mining"],
    choices=["partial", "full", "random", "msls_weighted"],
)



# Initialization parameters
parser.add_argument("--seed", type=int, default=train_args["seed"])
parser.add_argument(
    "--resume",
    type=str,
    default=train_args["resume"],
    help="Path to load checkpoint from, for resuming training or testing.",
)
# Other parameters
parser.add_argument("--device", type=str, default=train_args["device"], choices=["cuda", "cpu"])
parser.add_argument(
    "--num_workers", type=int, default=train_args["num_workers"], help="num_workers for all dataloaders"
)
parser.add_argument(
    "--resize",
    type=int,
    default=train_args["resize"],
    #default=[240, 320],
    nargs=2,
    help="Resizing shape for images (HxW).",
)
parser.add_argument(
    "--test_method",
    type=str,
    default=train_args["hard_resize"],
    choices=[
        "hard_resize",
        "single_query",
        "central_crop",
        "five_crops",
        "nearest_crop",
        "maj_voting",
    ],
    help="This includes pre/post-processing methods and prediction refinement",
)

parser.add_argument("--val_positive_dist_threshold", type=int, default=train_args["val_positive_dist_threshold"], help="_")
parser.add_argument(
    "--train_positives_dist_threshold", type=int, default=train_args["train_positives_dist_threshold"], help="_"
)
parser.add_argument(
    "--recall_values",
    type=int,
    default=train_args["recall_values"],
    nargs="+",
    help="Recalls to be computed, such as R@5.",
)
# Data augmentation parameters
parser.add_argument("--brightness", type=float, default=train_args["brightness"], help="_")
parser.add_argument("--contrast", type=float, default=train_args["contrast"], help="_")
parser.add_argument("--saturation", type=float, default=train_args["saturation"], help="_")
parser.add_argument("--hue", type=float, default=train_args["hue"], help="_")
parser.add_argument("--rand_perspective", type=float, default=train_args["rand_perspective"], help="_")
parser.add_argument("--horizontal_flip", action="store_true", help="_")
parser.add_argument("--random_resized_crop", type=float, default=train_args["random_resized_crop"], help="_")
parser.add_argument("--random_rotation", type=float, default=train_args["random_rotation"], help="_")
# Paths parameters
parser.add_argument(
    "--datasets_folder",
    type=str,
    default=train_args["datasets_folder"],
    help="Path with all datasets",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default=train_args["dataset_name"],
    help="Relative path of the dataset",
)

parser.add_argument(
    "--save_dir",
    type=str,
    default=train_args["save_dir"],
    help="Folder name of the current run (saved in ./logs/)",
)
args = parser.parse_args()
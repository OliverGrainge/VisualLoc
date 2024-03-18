import logging
import multiprocessing
import sys
from datetime import datetime

import torch

import PlaceRec.Training.EigenPlaces.commons as commons
import PlaceRec.Training.EigenPlaces.parser as parser
import PlaceRec.Training.EigenPlaces.test as test
from PlaceRec.Training.EigenPlaces.datasets.test_dataset import TestDataset

torch.backends.cudnn.benchmark = True  # Provides a speedup

args = parser.parse_arguments()
start_time = datetime.now()
output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)
commons.setup_logging(output_folder, console="info")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {output_folder}")
logging.info(
    f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs."
)

#### Model
if args.resume_model == "torchhub":
    model = torch.hub.load(
        "gmberton/eigenplaces",
        "get_trained_model",
        backbone=args.backbone,
        fc_output_dim=args.fc_output_dim,
    )

model = model.to(args.device)

test_ds = TestDataset(
    args.test_dataset_folder,
    queries_folder="queries",
    positive_dist_threshold=args.positive_dist_threshold,
)

recalls, recalls_str = test.test(args, test_ds, model)
logging.info(f"{test_ds}: {recalls_str}")

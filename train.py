import multiprocessing as mp

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from parsers import train_arguments
from PlaceRec.utils import get_config, get_method

args = train_arguments()

# ================================ GSVCities Training ==================================

if args.training_method == "gsv_cities_dense":
    from PlaceRec.Training.GSV_Cities.dense_trainer import dense_trainer

    trainer = dense_trainer

elif args.training_method == "gsv_cities_unstructured_sparse":
    from PlaceRec.Training.GSV_Cities.sparse_unstructured_trainer import (
        sparse_unstructured_trainer,
    )

    trainer = sparse_unstructured_trainer


elif args.training_method == "gsv_cities_semistructured_sparse":
    from PlaceRec.Training.GSV_Cities.sparse_semistructured_trainer import (
        sparse_semistructured_trainer,
    )

    trainer = sparse_semistructured_trainer


elif args.training_method == "gsv_cities_structured_sparse":
    from PlaceRec.Training.GSV_Cities.sparse_structured_trainer import (
        sparse_structured_trainer,
    )

    trainer = sparse_structured_trainer


elif args.training_method == "gsv_cities_distillation":
    from PlaceRec.Training.GSV_Cities.distillation_trainer import distillation_trainer

    trainer = distillation_trainer

elif args.training_method == "gsv_cities_sparse_structured_distillation":
    from PlaceRec.Training.GSV_Cities.stuctured_sparse_distillation_trainer import (
        sparse_structured_distillation_trainer,
    )

    trainer = sparse_structured_distillation_trainer

# ================================ Eigenplace Training ==================================
elif args.training_method == "eigenplaces":
    from PlaceRec.Training.EigenPlaces import train_eigenplaces

    trainer = train_eigenplaces


if __name__ == "__main__":
    mp.set_start_method("fork")
    trainer(args)

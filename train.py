import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from parsers import train_arguments
from PlaceRec.utils import get_config, get_method

args = train_arguments()

# ================================ GSVCities Training ==================================
if args.training_method == "gsv_cities_dense":
    from PlaceRec.Training.GSV_Cities.dense_trainer import dense_trainer

    dense_trainer(args)

elif args.training_method == "gsv_cities_unstructured_sparse":
    from PlaceRec.Training.GSV_Cities.sparse_unstructured_trainer import (
        sparse_unstructured_trainer,
    )

    sparse_unstructured_trainer(args)

elif args.training_method == "gsv_cities_semistructured_sparse":
    from PlaceRec.Training.GSV_Cities.sparse_semistructured_trainer import (
        sparse_semistructured_trainer,
    )

    sparse_semistructured_trainer(args)

elif args.training_method == "gsv_cities_structured_sparse":
    from PlaceRec.Training.GSV_Cities.sparse_structured_trainer import (
        sparse_structured_trainer,
    )

    sparse_structured_trainer(args)


# ================================ Eigenplace Training ==================================
elif args.training_method == "eigenplaces":
    from PlaceRec.Training.EigenPlaces import train_eigenplaces

    method = get_method(args.method, args.pretrained)
    model = method.model
    model.train()
    train_eigenplaces(model, features_dim=method.features_dim)


elif args.training_method == "cosplace":
    raise NotImplementedError

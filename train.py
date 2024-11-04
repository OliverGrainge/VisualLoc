import multiprocessing as mp
from parsers import train_arguments

args = train_arguments()


if args.training_method == "gsv_cities_dense":
    from PlaceRec.Training.GSV_Cities.dense_trainer import dense_trainer

    trainer = dense_trainer

elif args.training_method == "gsv_cities_structured_sparse":
    from PlaceRec.Training.GSV_Cities.sparse_structured_trainer import (
        sparse_structured_trainer,
    )
    trainer = sparse_structured_trainer


if __name__ == "__main__":
    mp.set_start_method("fork")
    trainer(args)

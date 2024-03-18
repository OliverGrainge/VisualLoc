import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from parsers import train_arguments
from PlaceRec.utils import get_config, get_method

args = train_arguments()
method = get_method(args.method, args.pretrained)

if args.training_method == "gsv_cities":
    from PlaceRec.Training.GSV_Cities.dataloaders.GSVCitiesDataloader import (
        GSVCitiesDataModule,
    )
    from PlaceRec.Training.GSV_Cities.trainer import VPRModel

    pl.seed_everything(seed=1, workers=True)
    torch.set_float32_matmul_precision("medium")

    datamodule = GSVCitiesDataModule(
        batch_size=int(args.batch_size / 4),
        img_per_place=4,
        min_img_per_place=4,
        # cities=['London', 'Boston', 'Melbourne'], # you can sppecify cities here or in GSVCitiesDataloader.py
        shuffle_all=False,  # shuffle all images or keep shuffling in-city only
        random_sample_from_each_place=True,
        image_size=args.image_resolution,
        num_workers=8,
        show_data_stats=False,
        val_set_names=["pitts30k_val"],  # pitts30k_val
    )

    model = VPRModel(
        method=method,
        lr=0.0002,  # 0.03 for sgd
        optimizer="adam",  # sgd, adam or adamw
        weight_decay=0,  # 0.001 for sgd or 0.0 for adam
        momentum=0.9,
        warmup_steps=600,
        milestones=[5, 10, 15, 25],
        lr_mult=0.3,
        # ---------------------------------
        # ---- Training loss function -----
        loss_name="MultiSimilarityLoss",
        miner_name="MultiSimilarityMiner",  # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.1,
        faiss_gpu=False,
    )

    # model params saving using Pytorch Lightning
    checkpoint_cb = ModelCheckpoint(
        dirpath="Checkpoints/gsv_cities/",
        monitor="pitts30k_val/R1",
        filename=f"{method.name}"
        + "_epoch({epoch:02d})_step({step:04d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]",
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        mode="max",
    )

    # ------------------
    # we instantiate a trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        default_root_dir=f"./LOGS/{method.name}",  # Tensorflow can be used to viz
        num_sanity_val_steps=0,  # runs N validation steps before stating training
        precision="16-mixed",  # we use half precision to reduce  memory usage (and 2x speed on RTX)
        max_epochs=30,
        check_val_every_n_epoch=1,  # run validation every epoch
        callbacks=[
            checkpoint_cb
        ],  # we run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1,  # we reload the dataset to shuffle the order
        log_every_n_steps=20,
        limit_train_batches=100,
        # fast_dev_run=True # comment if you want to start training the network and saving checkpoints
    )

    # we call the trainer, and give it the model and the datamodule
    # now you see the modularity of Pytorch Lighning?
    trainer.fit(model=model, datamodule=datamodule)


elif args.training_method == "eigenplaces":
    from PlaceRec.Training.EigenPlaces import train_eigenplaces

    method = get_method(args.method, args.pretrained)
    train_eigenplaces(method.model, features_dim=method.features_dim)

elif args.training_method == "cosplace":
    raise NotImplementedError

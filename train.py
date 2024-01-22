from train import GSVCitiesDataModule, VPRModel
from PlaceRec.utils import get_method
from parsers import train_arguments
import os 
from os.path import join 
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import ModelCheckpoint
from PlaceRec.utils import get_config, get_training_logger
from compute_features import TRAIN_CITIES


config = get_config()
args = train_arguments()
args.cities = TRAIN_CITIES
method = get_method(args.method, pretrained=False)
args.features_dim = method.features_dim
model = method.model.to(args.device)



datamodule = GSVCitiesDataModule(args)
model = VPRModel(args, model)

# Checkpointing the Model
checkpoint_callback = ModelCheckpoint(
    monitor="pitts30k_val/R1",
    filename=join(
        os.getcwd(),
        "PlaceRec/Training/Contrastive/checkpoints/",
        method.name,
        f"{method.name}" + "_epoch({epoch:02d})_step({step:04d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]",
    ),
    auto_insert_metric_name=False,
    save_top_k=1,
    verbose=False,
    mode="max",
)

logger = get_training_logger(config, project_name="MultiTeacherDistillation")

trainer = pl.Trainer(
    accelerator="gpu",
    devices=[0],
    default_root_dir=f"/Logs/{method.name}",  # Tensorflow can be used to viz
    num_sanity_val_steps=0,  # runs N validation steps before stating training
    #precision=16,  # we use half precision to reduce  memory usage (and 2x speed on RTX)
    max_epochs=30,
    check_val_every_n_epoch=1,  # run validation every epoch
    callbacks=[checkpoint_callback],  # we run the checkpointing callback (you can add more)
    logger=logger,
    reload_dataloaders_every_n_epochs=1,  # we reload the dataset to shuffle the order
    log_every_n_steps=20,
    val_check_interval=400
    #limit_train_batches=10,
    # fast_dev_run=True # comment if you want to start training the network and saving checkpoints
)
trainer.fit(model=model, datamodule=datamodule)

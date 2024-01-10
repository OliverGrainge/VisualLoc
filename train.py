import os

from PlaceRec.utils import get_method, get_training_logger
import pytorch_lightning as pl 
from PlaceRec.utils import get_config
from os.path import join
from Distillation import DistillationDataModule, DistillationModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from parsers import train_args

config = get_config()
args = train_args()


student_method = get_method(args.student_method, pretrained=False)
teacher_method = get_method(args.teacher_method, pretrained=True)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00,
    patience=args.patience,
    verbose=False,
    mode="min",
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=os.path.join(os.getcwd(), "Distillation/checkpoints/", teacher_method.name),
    filename=student_method.name + "_" + args.distillation_type + "_" + str(args.size[0]) + "-" + str(args.size[1]) + "-{epoch:02d}-{val_loss:.5f}",
    save_top_k=1,
    verbose=False,
    mode="min",
)


logger = get_training_logger(config, project_name="Distillation")

trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    accelerator="gpu" if args.device in ["mps", "cuda"] else "cpu",
    logger=logger,
    callbacks=[early_stop_callback, checkpoint_callback],
    val_check_interval=500
)

if __name__ == "__main__":
    distillationdatamodule = DistillationDataModule(args, teacher_method, teacher_method.preprocess, reload=args.reload)
    distillationmodule = DistillationModule(args, student_method)
    trainer.fit(distillationmodule, datamodule=distillationdatamodule)
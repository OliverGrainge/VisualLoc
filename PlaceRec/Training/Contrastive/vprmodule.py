
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
from PlaceRec.Training.Contrastive import utils
from PlaceRec.Training.Contrastive.dataloaders.GSVCitiesDataloader import GSVCitiesDataModule



class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.
    """

    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = utils.get_loss(args.loss_name)
        self.miner = utils.get_miner(args.miner_name, args.miner_margin)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.faiss_gpu = args.faiss_gpu

        self.model = model
        
    # the forward pass of the lightning model
    def forward(self, x):
        x = self.model(x)
        return x
    
    # configure the optimizer 
    def configure_optimizers(self):
        if self.args.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), 
                                        lr=self.args.learning_rate, 
                                        weight_decay=self.args.weight_decay, 
                                        momentum=self.args.momentum)
        elif self.args.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.args.learning_rate, 
                                        weight_decay=self.args.weight_decay)
        elif self.args.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.args.learning_rate, 
                                        weight_decay=self.args.weight_decay)
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones, gamma=self.args.lr_mult)
        warmup_scheduler = {
            'scheduler': LambdaLR(optimizer, lr_lambda=lambda epoch: min(1.0, (epoch + 1) / self.args.warmup_steps)),
            'interval': 'step',  # Step-wise scheduler
        }
        return [optimizer], [warmup_scheduler, scheduler]

        
    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
            
            # calculate the % of trivial pairs/triplets 
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)

        else: # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple: 
                # somes losses do the online mining inside (they don't need a miner objet), 
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class, 
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log('b_acc', sum(self.batch_acc) /
                len(self.batch_acc), prog_bar=True, logger=True)
        return loss
    
    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        places, labels = batch
        
        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape
        
        # reshape places and labels
        images = places.view(BS*N, ch, h, w)
        labels = labels.view(-1)

        # Feed forward the batch to the model
        descriptors = self(images) # Here we are calling the method forward that we defined above
        loss = self.loss_function(descriptors, labels) # Call the loss_function we defined above
        
        self.log('loss', loss.item(), logger=True)
        return {'loss': loss}
    
    # This is called at the end of eatch training epoch
    def on_train_epoch_end(self):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def on_validation_epoch_start(self):
        if len(self.trainer.datamodule.val_set_names) == 1:
            self.val_step_outputs = []
        else:
            self.val_step_outputs = [[] for _ in range(len(self.trainer.datamodule.val_set_names))]

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        # calculate descriptors
        descriptors = self(places).detach().cpu()
        if len(self.trainer.datamodule.val_set_names) == 1:
            self.val_step_outputs.append(descriptors)
        else: 
            self.val_step_outputs[dataloader_idx].append(descriptors)
        return descriptors
    
    def on_validation_epoch_end(self):
        """at the end of each validation epoch
        descriptors are returned in their order
        depending on how the validation dataset is implemented 
        for this project (MSLS val, Pittburg val), it is always references then queries.
        For example, if we have n references and m queries, we will get 
        the descriptors for each val_dataset in a list as follows: 
        [R1, R2, ..., Rn, Q1, Q2, ..., Qm]
        we then split it to references=[R1, R2, ..., Rn] and queries=[Q1, Q2, ..., Qm]
        to calculate recall@K using the ground truth provided.
        """
        val_step_outputs = self.val_step_outputs
        self.val_step_outputs = []
        dm = self.trainer.datamodule
        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
        if len(dm.val_datasets)==1: # we need to put the outputs in a list
            val_step_outputs = [val_step_outputs]
        
        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)
            
            num_references = val_dataset.num_references
            num_queries = val_dataset.num_queries
            ground_truth = val_dataset.ground_truth
            
            # split to ref and queries    
            r_list = feats[ : num_references]
            q_list = feats[num_references : ]

            recalls_dict, predictions = utils.get_validation_recalls(r_list=r_list, 
                                                q_list=q_list,
                                                k_values=[1, 5, 10, 15, 20, 25],
                                                gt=ground_truth,
                                                print_results=True,
                                                dataset_name=val_set_name,
                                                faiss_gpu=self.args.faiss_gpu
                                                )
            del r_list, q_list, feats, num_references, ground_truth
            self.log(f'{val_set_name}/R1', recalls_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', recalls_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', recalls_dict[10], prog_bar=False, logger=True)
        print('\n\n')
            
            

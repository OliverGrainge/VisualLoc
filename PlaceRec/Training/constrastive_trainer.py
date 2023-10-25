import math

import contrastive_dataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import get_loss_function


class ConstrastiveTrainer:
    def ___init__(self, args, method):
        self.args = args
        self.model = method.model
        self.train_dataset = contrastive_dataset.TripletsDataset(args, args.datasets_folder, args.dataset_name, "train", args.negs_num_per_query)
        self.val_dataset = contrastive_dataset.BaseDataset(args, args.datasets_folder, args.dataset_name, "val")
        self.test_dataset = test_ds = contrastive_dataset.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
        self.loss_fn = get_loss_function(args)
        self.optimizer = get_optimizer(args, self.model)

        self.model.cuda()

    def fit():
        pass

    def train_one_epoch(self) -> None:
        loops_num = math.ceil(self.args.queries_per_epoch / self.args.cache_refresh_rate)
        for loop in range(loops_num):
            self.train_dataset.is_inference = True
            self.train_ds.compute_triplets(self.args, self.model)
            self.train_dataset.is_inference = False

            train_dataloader = DataLoader(
                dataset=self.train_dataset,
                num_workers=self.args.num_workers,
                batch_size=self.args.train_batch_size,
                collate_fn=self.train_dataset.collate_fn,
                pin_memory=(self.args.device == "cuda"),
                drop_last=True,
            )

            self.model = self.model.train()

            for images, triplets_local_indexes, _ in tqdm(train_dataloader, ncols=100):
                features = self.model(images.to(self.args.device))

                loss_triplet = 0

                if self.args.criterion == "triplet":
                    triplets_local_indexes = torch.transpose(
                        triplets_local_indexes.view(self.args.train_batch_size, self.args.negs_num_per_query, 3), 1, 0
                    )
                    for triplets in triplets_local_indexes:
                        queries_indexes, positives_indexes, negatives_indexes = triplets.T
                        loss_triplet += self.loss_fn(features[queries_indexes], features[positives_indexes], features[negatives_indexes])
                elif self.args.criterion == "sare_joint":
                    # sare_joint needs to receive all the negatives at once
                    triplet_index_batch = triplets_local_indexes.view(args.train_batch_size, 10, 3)
                    for batch_triplet_index in triplet_index_batch:
                        q = features[batch_triplet_index[0, 0]].unsqueeze(0)  # obtain query as tensor of shape 1xn_features
                        p = features[batch_triplet_index[0, 1]].unsqueeze(0)  # obtain positive as tensor of shape 1xn_features
                        n = features[batch_triplet_index[:, 2]]  # obtain negatives as tensor of shape 10xn_features
                        loss_triplet += self.loss_fn(q, p, n)
                elif self.args.criterion == "sare_ind":
                    for triplet in triplets_local_indexes:
                        # triplet is a 1-D tensor with the 3 scalars indexes of the triplet
                        q_i, p_i, n_i = triplet
                        loss_triplet += self.loss_fn(features[q_i : q_i + 1], features[p_i : p_i + 1], features[n_i : n_i + 1])

            del features
            loss_triplet /= self.args.train_batch_size * self.args.negs_num_per_query

            loss_triplet.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            # Keep track of all losses by appending them to epoch_losses
            batch_loss = loss_triplet.item()
            epoch_losses = np.append(epoch_losses, batch_loss)
            del loss_triplet

    def validate():
        pass 

    def checkpoint_model():
        pass 

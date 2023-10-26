import math
import shutil
import time
from collections import OrderedDict
from os.path import join

import contrastive_dataset
import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm
from utils import get_loss_function, get_optimizer


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

    def fit(self):
        epoch_st = time.time()

        best_r5, start_epoch_num, not_improved_num = 0, 0, 0
        if self.args.resume is not None:
            best_r5, start_epoch_num, not_improved_num = self.resume_train()

        self.model.to(self.args.device)

        for epoch_num in range(start_epoch_num, self.args.epochs_num):
            self.train_one_epoch()
            recalls, recalls_str = self.test()
            print(recalls_str)
            is_best = recalls[1] > best_r5
            self.save_checkpoint(
                self.args,
                {
                    "epoch_num": epoch_num,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "recalls": recalls,
                    "best_r5": best_r5,
                    "not_improved_num": not_improved_num,
                },
                is_best,
                filename="last_model.pth",
            )

            if is_best:
                print(f"Improved: previous best R@5 = {best_r5:.1f}, current R@5 = {recalls[1]:.1f}")
                best_r5 = recalls[1]
                not_improved_num = 0
            else:
                not_improved_num += 1
                print(f"Not improved: {not_improved_num} / {self.args.patience}: best R@5 = {best_r5:.1f}, current R@5 = {recalls[1]:.1f}")
                if not_improved_num >= self.args.patience:
                    print(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
                    break

        recalls, recalls_str = self.test(test_method=self.args.test_method)
        print(f"Recalls on {self.test_dataset}: {recalls_str}")

    def train_one_epoch(self) -> None:
        epoch_losses = np.zeros((0, 1), dtype=np.float32)
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

    def test(self, test_method: str = "hard_resize"):
        """Compute features of the given dataset and compute the recalls."""

        assert test_method in [
            "hard_resize",
            "single_query",
            "central_crop",
            "five_crops",
            "nearest_crop",
            "maj_voting",
        ], f"test_method can't be {test_method}"

        model = model.eval()
        with torch.no_grad():
            # For database use "hard_resize", although it usually has no effect because database images have same resolution
            self.val_dataset.test_method = "hard_resize"
            database_subset_ds = Subset(self.val_dataset, list(range(self.val_dataset.database_num)))
            database_dataloader = DataLoader(
                dataset=database_subset_ds,
                num_workers=self.args.num_workers,
                batch_size=self.args.infer_batch_size,
                pin_memory=(self.args.device == "cuda"),
                drop_last=True,
            )

            if test_method == "nearest_crop" or test_method == "maj_voting":
                all_features = np.empty((5 * self.val_dataset.queries_num + self.val_dataset.database_num, args.features_dim), dtype="float32")
            else:
                all_features = np.empty((len(self.val_dataset), self.args.features_dim), dtype="float32")

            for inputs, indices in tqdm(database_dataloader, ncols=100):
                if self.args.precision == "mixed" or self.args.precision == "fp16":
                    inputs = inputs.half()
                with torch.no_grad():
                    features = model(inputs.to(sef.args.device)).float()
                if torch.isnan(features).any():
                    raise Exception("Features have NAN in them")
                features = features.cpu().numpy()
                all_features[indices.numpy(), :] = features

            queries_infer_batch_size = 1 if test_method == "single_query" else self.args.infer_batch_size
            self.val_dataset.test_method = test_method
            queries_subset_ds = Subset(
                self.val_dataset, list(range(self.val_dataset.database_num, self.val_dataset.database_num + self.val_dataset.queries_num))
            )
            queries_dataloader = DataLoader(
                dataset=queries_subset_ds,
                num_workers=self.args.num_workers,
                batch_size=queries_infer_batch_size,
                pin_memory=(self.args.device == "cuda"),
                drop_last=True,
            )
            for inputs, indices in tqdm(queries_dataloader, ncols=100):
                if test_method == "five_crops" or test_method == "nearest_crop" or test_method == "maj_voting":
                    inputs = torch.cat(tuple(inputs))  # shape = 5*bs x 3 x 480 x 480

                if self.args.precision == "mixed" or self.args.precision == "fp16":
                    inputs = inputs.half()

                features = model(inputs.to(self.args.device)).float()
                if test_method == "five_crops":  # Compute mean along the 5 crops
                    features = torch.stack(torch.split(features, 5)).mean(1)
                features = features.cpu().numpy()

                if test_method == "nearest_crop" or test_method == "maj_voting":  # store the features of all 5 crops
                    start_idx = self.val_dataset.database_num + (indices[0] - self.val_dataset.database_num) * 5
                    end_idx = start_idx + indices.shape[0] * 5
                    indices = np.arange(start_idx, end_idx)
                    all_features[indices, :] = features
                else:
                    all_features[indices.numpy(), :] = features

        queries_features = all_features[self.val_dataset.database_num :]
        database_features = all_features[: self.val_dataset.database_num]

        faiss_index = faiss.IndexFlatL2(self.args.features_dim)
        faiss_index.add(database_features)
        del database_features, all_features

        distances, predictions = faiss_index.search(queries_features, max(self.args.recall_values))

        if test_method == "nearest_crop":
            distances = np.reshape(distances, (self.val_dataset.queries_num, 20 * 5))
            predictions = np.reshape(predictions, (self.val_dataset.queries_num, 20 * 5))
            for q in range(self.val_dataset.queries_num):
                # sort predictions by distance
                sort_idx = np.argsort(distances[q])
                predictions[q] = predictions[q, sort_idx]
                # remove duplicated predictions, i.e. keep only the closest ones
                _, unique_idx = np.unique(predictions[q], return_index=True)
                # unique_idx is sorted based on the unique values, sort it again
                predictions[q, :20] = predictions[q, np.sort(unique_idx)][:20]
            predictions = predictions[:, :20]  # keep only the closer 20 predictions for each query
        elif test_method == "maj_voting":
            distances = np.reshape(distances, (self.val_dataset.queries_num, 5, 20))
            predictions = np.reshape(predictions, (self.val_dataset.queries_num, 5, 20))
            for q in range(self.val_dataset.queries_num):
                # votings, modify distances in-place
                self.top_n_voting("top1", predictions[q], distances[q], self.args.majority_weight)
                self.argstop_n_voting("top5", predictions[q], distances[q], self.args.majority_weight)
                self.top_n_voting("top10", predictions[q], distances[q], self.args.majority_weight)

                # flatten dist and preds from 5, 20 -> 20*5
                # and then proceed as usual to keep only first 20
                dists = distances[q].flatten()
                preds = predictions[q].flatten()

                # sort predictions by distance
                sort_idx = np.argsort(dists)
                preds = preds[sort_idx]
                # remove duplicated predictions, i.e. keep only the closest ones
                _, unique_idx = np.unique(preds, return_index=True)
                # unique_idx is sorted based on the unique values, sort it again
                # here the row corresponding to the first crop is used as a
                # 'buffer' for each query, and in the end the dimension
                # relative to crops is eliminated
                predictions[q, 0, :20] = preds[np.sort(unique_idx)][:20]
            predictions = predictions[:, 0, :20]  # keep only the closer 20 predictions for each query

        #### For each query, check if the predictions are correct
        positives_per_query = self.val_dataset.get_positives()
        # args.recall_values by default is [1, 5, 10, 20]
        recalls = np.zeros(len(self.args.recall_values))
        for query_index, pred in enumerate(predictions):
            for i, n in enumerate(self.args.recall_values):
                if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                    recalls[i:] += 1
                    break
        # Divide by the number of queries*100, so the recalls are in percentages
        recalls = recalls / self.val_dataset.queries_num * 100
        recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(self.args.recall_values, recalls)])
        return recalls, recalls_str

    def validate(self, test_method: str = "hard_resize"):
        """Compute features of the given dataset and compute the recalls."""

        assert test_method in [
            "hard_resize",
            "single_query",
            "central_crop",
            "five_crops",
            "nearest_crop",
            "maj_voting",
        ], f"test_method can't be {test_method}"

        model = model.eval()
        with torch.no_grad():
            # For database use "hard_resize", although it usually has no effect because database images have same resolution
            self.test_dataset.test_method = "hard_resize"
            database_subset_ds = Subset(self.test_dataset, list(range(self.val_dataset.database_num)))
            database_dataloader = DataLoader(
                dataset=database_subset_ds,
                num_workers=self.args.num_workers,
                batch_size=self.args.infer_batch_size,
                pin_memory=(self.args.device == "cuda"),
                drop_last=True,
            )

            if test_method == "nearest_crop" or test_method == "maj_voting":
                all_features = np.empty((5 * self.test_dataset.queries_num + self.val_dataset.database_num, args.features_dim), dtype="float32")
            else:
                all_features = np.empty((len(self.test_dataset), self.args.features_dim), dtype="float32")

            for inputs, indices in tqdm(database_dataloader, ncols=100):
                if self.args.precision == "mixed" or self.args.precision == "fp16":
                    inputs = inputs.half()
                with torch.no_grad():
                    features = model(inputs.to(sef.args.device)).float()
                if torch.isnan(features).any():
                    raise Exception("Features have NAN in them")
                features = features.cpu().numpy()
                all_features[indices.numpy(), :] = features

            queries_infer_batch_size = 1 if test_method == "single_query" else self.args.infer_batch_size
            self.test_dataset.test_method = test_method
            queries_subset_ds = Subset(
                self.test_dataset, list(range(self.test_dataset.database_num, self.test_dataset.database_num + self.test_dataset.queries_num))
            )
            queries_dataloader = DataLoader(
                dataset=queries_subset_ds,
                num_workers=self.args.num_workers,
                batch_size=queries_infer_batch_size,
                pin_memory=(self.args.device == "cuda"),
                drop_last=True,
            )
            for inputs, indices in tqdm(queries_dataloader, ncols=100):
                if test_method == "five_crops" or test_method == "nearest_crop" or test_method == "maj_voting":
                    inputs = torch.cat(tuple(inputs))  # shape = 5*bs x 3 x 480 x 480

                if self.args.precision == "mixed" or self.args.precision == "fp16":
                    inputs = inputs.half()

                features = model(inputs.to(self.args.device)).float()
                if test_method == "five_crops":  # Compute mean along the 5 crops
                    features = torch.stack(torch.split(features, 5)).mean(1)
                features = features.cpu().numpy()

                if test_method == "nearest_crop" or test_method == "maj_voting":  # store the features of all 5 crops
                    start_idx = self.test_dataset.database_num + (indices[0] - self.test_dataset.database_num) * 5
                    end_idx = start_idx + indices.shape[0] * 5
                    indices = np.arange(start_idx, end_idx)
                    all_features[indices, :] = features
                else:
                    all_features[indices.numpy(), :] = features

        queries_features = all_features[self.test_dataset.database_num :]
        database_features = all_features[: self.test_dataset.database_num]

        faiss_index = faiss.IndexFlatL2(self.args.features_dim)
        faiss_index.add(database_features)
        del database_features, all_features

        distances, predictions = faiss_index.search(queries_features, max(self.args.recall_values))

        if test_method == "nearest_crop":
            distances = np.reshape(distances, (self.test_dataset.queries_num, 20 * 5))
            predictions = np.reshape(predictions, (self.test_dataset.queries_num, 20 * 5))
            for q in range(self.test_dataset.queries_num):
                # sort predictions by distance
                sort_idx = np.argsort(distances[q])
                predictions[q] = predictions[q, sort_idx]
                # remove duplicated predictions, i.e. keep only the closest ones
                _, unique_idx = np.unique(predictions[q], return_index=True)
                # unique_idx is sorted based on the unique values, sort it again
                predictions[q, :20] = predictions[q, np.sort(unique_idx)][:20]
            predictions = predictions[:, :20]  # keep only the closer 20 predictions for each query
        elif test_method == "maj_voting":
            distances = np.reshape(distances, (self.test_dataset.queries_num, 5, 20))
            predictions = np.reshape(predictions, (self.test_dataset.queries_num, 5, 20))
            for q in range(self.test_dataset.queries_num):
                # votings, modify distances in-place
                self.top_n_voting("top1", predictions[q], distances[q], self.args.majority_weight)
                self.argstop_n_voting("top5", predictions[q], distances[q], self.args.majority_weight)
                self.top_n_voting("top10", predictions[q], distances[q], self.args.majority_weight)

                # flatten dist and preds from 5, 20 -> 20*5
                # and then proceed as usual to keep only first 20
                dists = distances[q].flatten()
                preds = predictions[q].flatten()

                # sort predictions by distance
                sort_idx = np.argsort(dists)
                preds = preds[sort_idx]
                # remove duplicated predictions, i.e. keep only the closest ones
                _, unique_idx = np.unique(preds, return_index=True)
                # unique_idx is sorted based on the unique values, sort it again
                # here the row corresponding to the first crop is used as a
                # 'buffer' for each query, and in the end the dimension
                # relative to crops is eliminated
                predictions[q, 0, :20] = preds[np.sort(unique_idx)][:20]
            predictions = predictions[:, 0, :20]  # keep only the closer 20 predictions for each query

        #### For each query, check if the predictions are correct
        positives_per_query = self.test_dataset.get_positives()
        # args.recall_values by default is [1, 5, 10, 20]
        recalls = np.zeros(len(self.args.recall_values))
        for query_index, pred in enumerate(predictions):
            for i, n in enumerate(self.args.recall_values):
                if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                    recalls[i:] += 1
                    break
        # Divide by the number of queries*100, so the recalls are in percentages
        recalls = recalls / self.test_dataset.queries_num * 100
        recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(self.args.recall_values, recalls)])
        return recalls, recalls_str

    def top_n_voting(self, topn, predictions, distances, maj_weight):
        if topn == "top1":
            n = 1
            selected = 0
        elif topn == "top5":
            n = 5
            selected = slice(0, 5)
        elif topn == "top10":
            n = 10
            selected = slice(0, 10)
        # find predictions that repeat in the first, first five,
        # or fist ten columns for each crop
        vals, counts = np.unique(predictions[:, selected], return_counts=True)
        # for each prediction that repeats more than once,
        # subtract from its score
        for val, count in zip(vals[counts > 1], counts[counts > 1]):
            mask = predictions[:, selected] == val
            distances[:, selected][mask] -= maj_weight * count / n

    def save_checkpoint(args, state, is_best, filename):
        model_path = join(args.save_dir, filename)
        torch.save(state, model_path)
        if is_best:
            shutil.copyfile(model_path, join(args.save_dir, "best_model.pth"))

    def resume_model(self):
        checkpoint = torch.load(self.args.resume, map_location=self.args.device)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            # The pre-trained models that we provide in the README do not have 'state_dict' in the keys as
            # the checkpoint is directly the state dict
            state_dict = checkpoint
        # if the model contains the prefix "module" which is appendend by
        # DataParallel, remove it to avoid errors when loading dict
        if list(state_dict.keys())[0].startswith("module"):
            state_dict = OrderedDict({k.replace("module.", ""): v for (k, v) in state_dict.items()})

        for i, layer_name in enumerate(list(state_dict.keys())):
            if layer_name != list(self.model.state_dict().keys())[i]:
                print(layer_name != list(self.model.state_dict().keys())[i])
        self.model.load_state_dict(state_dict)

    def resume_train(self, strict=False):
        """Load model, optimizer, and other training parameters"""
        checkpoint = torch.load(args.resume)
        start_epoch_num = checkpoint["epoch_num"]
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_r5 = checkpoint["best_r5"]
        not_improved_num = checkpoint["not_improved_num"]
        if self.args.resume.endswith("last_model.pth"):  # Copy best model to current save_dir
            shutil.copy(self.args.resume.replace("last_model.pth", "best_model.pth"), self.args.save_dir)
        return best_r5, start_epoch_num, not_improved_num

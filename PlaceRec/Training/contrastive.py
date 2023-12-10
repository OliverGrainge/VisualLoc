
import os
from glob import glob
from os.path import join

import faiss
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm
from PlaceRec.utils import ImageIdxDataset
from os.path import join
import pytorch_lightning as pl
from PlaceRec.utils import get_loss_function
from torch import optim

base_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def path_to_pil_img(path):
    return Image.open(path).convert("RGB")


def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (images,
        triplets_local_indexes, triplets_global_indexes).
        triplets_local_indexes are the indexes referring to each triplet within images.
        triplets_global_indexes are the global indexes of each image.
    Args:
        batch: list of tuple (images, triplets_local_indexes, triplets_global_indexes).
            considering each query to have 10 negatives (negs_num_per_query=10):
            - images: torch tensor of shape (12, 3, h, w).
            - triplets_local_indexes: torch tensor of shape (10, 3).
            - triplets_global_indexes: torch tensor of shape (12).
    Returns:
        images: torch tensor of shape (batch_size*12, 3, h, w).
        triplets_local_indexes: torch tensor of shape (batch_size*10, 3).
        triplets_global_indexes: torch tensor of shape (batch_size, 12).
    """
    images = torch.cat([e[0] for e in batch])
    triplets_local_indexes = torch.cat([e[1][None] for e in batch])
    triplets_global_indexes = torch.cat([e[2][None] for e in batch])
    for i, (local_indexes, global_indexes) in enumerate(
        zip(triplets_local_indexes, triplets_global_indexes)
    ):
        local_indexes += (
            len(global_indexes) * i
        )  # Increment local indexes by offset (len(global_indexes) is 12)
    return images, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes


class PCADataset(data.Dataset):
    def __init__(
        self, args, datasets_folder="dataset", dataset_folder="pitts30k/images/train"
    ):
        dataset_folder_full_path = join(datasets_folder, dataset_folder)
        if not os.path.exists(dataset_folder_full_path):
            raise FileNotFoundError(f"Folder {dataset_folder_full_path} does not exist")
        self.images_paths = sorted(
            glob(join(dataset_folder_full_path, "**", "*.jpg"), recursive=True)
        )

    def __getitem__(self, index):
        return base_transform(path_to_pil_img(self.images_paths[index]))

    def __len__(self):
        return len(self.images_paths)


class BaseDataset(data.Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache)."""

    def __init__(
        self, args, test_preprocess, datasets_folder="datasets", dataset_name="pitts30k", split="train"
    ):
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.dataset_folder = join(datasets_folder, dataset_name, "images", split)
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")

        self.test_preprocess = test_preprocess

        #### Read paths and UTM coordinates for all images.
        database_folder = join(self.dataset_folder, "database")
        queries_folder = join(self.dataset_folder, "queries")

        if not os.path.exists(database_folder):
            raise FileNotFoundError(f"Folder {database_folder} does not exist")
        if not os.path.exists(queries_folder):
            raise FileNotFoundError(f"Folder {queries_folder} does not exist")

        self.database_paths = sorted(
            glob(join(database_folder, "**", "*.jpg"), recursive=True)
        )
        self.queries_paths = sorted(
            glob(join(queries_folder, "**", "*.jpg"), recursive=True)
        )

        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array(
            [(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]
        ).astype(float)
        self.queries_utms = np.array(
            [(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]
        ).astype(float)

        # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.soft_positives_per_query = knn.radius_neighbors(
            self.queries_utms,
            radius=args.val_positive_dist_threshold,
            return_distance=False,
        )

        self.images_paths = list(self.database_paths) + list(self.queries_paths)
        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)

    def __getitem__(self, index):
        img = path_to_pil_img(self.images_paths[index])
        img = self.test_preprocess(img)
        return img, index

    def _test_query_transform(self, img):
        """Transform query image according to self.test_method."""
        C, H, W = img.shape
        if self.test_method == "single_query":
            # self.test_method=="single_query" is used when queries have varying sizes, and can't be stacked in a batch.
            processed_img = T.functional.resize(img, min(self.resize), antialias=True)
        elif self.test_method == "central_crop":
            # Take the biggest central crop of size self.resize. Preserves ratio.
            scale = max(self.resize[0] / H, self.resize[1] / W)
            processed_img = torch.nn.functional.interpolate(
                img.unsqueeze(0), scale_factor=scale
            ).squeeze(0)
            processed_img = T.functional.center_crop(processed_img, self.resize)
            assert processed_img.shape[1:] == torch.Size(
                self.resize
            ), f"{processed_img.shape[1:]} {self.resize}"
        elif (
            self.test_method == "five_crops"
            or self.test_method == "nearest_crop"
            or self.test_method == "maj_voting"
        ):
            # Get 5 square crops with size==shorter_side (usually 480). Preserves ratio and allows batches.
            shorter_side = min(self.resize)
            processed_img = T.functional.resize(img, shorter_side)
            processed_img = torch.stack(
                T.functional.five_crop(processed_img, shorter_side)
            )
            assert processed_img.shape == torch.Size(
                [5, 3, shorter_side, shorter_side]
            ), f"{processed_img.shape} {torch.Size([5, 3, shorter_side, shorter_side])}"
        return processed_img

    def __len__(self):
        return len(self.images_paths)

    def __repr__(self):
        return f"< {self.__class__.__name__}, {self.dataset_name} - #database: {self.database_num}; #queries: {self.queries_num} >"

    def get_positives(self):
        return self.soft_positives_per_query


class TripletsDataset(BaseDataset):
    """Dataset used for training, it is used to compute the triplets
    with TripletsDataset.compute_triplets() with various mining methods.
    If is_inference == True, uses methods of the parent class BaseDataset,
    this is used for example when computing the cache, because we compute features
    of each image, not triplets.
    """

    def __init__(
        self,
        args,
        train_preprocess,
        test_preprocess,
        split="train",
    ):
        super().__init__(args, test_preprocess, args.datasets_folder, args.dataset_name, split)
        self.mining = args.mining
        self.neg_num_per_query = (args.neg_num_per_query)

        if (
            self.args.mining == "full"
        ):  # "Full database mining" keeps a cache with last used negatives
            self.neg_cache = [
                np.empty((0,), dtype=np.int32) for _ in range(self.queries_num)
            ]
        self.is_inference = False

        self.train_preprocess = train_preprocess
        self.test_preprocess = test_preprocess

        # Find hard_positives_per_query, which are within train_positives_dist_threshold (10 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.hard_positives_per_query = list(
            knn.radius_neighbors(
                self.queries_utms,
                radius=args.train_positive_dist_threshold,  # 10 meters
                return_distance=False,
            )
        )

        #### Some queries might have no positive, we should remove those queries.
        queries_without_any_hard_positive = np.where(
            np.array([len(p) for p in self.hard_positives_per_query], dtype=object) == 0
        )[0]
        if len(queries_without_any_hard_positive) != 0:
            print(
                f"There are {len(queries_without_any_hard_positive)} queries without any positives "
                + "within the training set. They won't be considered as they're useless for training."
            )
        # Remove queries without positives
        self.hard_positives_per_query = [
            arr
            for i, arr in enumerate(self.hard_positives_per_query)
            if i not in queries_without_any_hard_positive
        ]

        self.queries_paths = [
            arr
            for i, arr in enumerate(self.queries_paths)
            if i not in queries_without_any_hard_positive
        ]

        # Recompute images_paths and queries_num because some queries might have been removed
        self.images_paths = list(self.database_paths) + list(self.queries_paths)
        self.queries_num = len(self.queries_paths)

        # msls_weighted refers to the mining presented in MSLS paper's supplementary.
        # Basically, images from uncommon domains are sampled more often. Works only with MSLS dataset.
        if self.mining == "msls_weighted":
            notes = [p.split("@")[-2] for p in self.queries_paths]
            try:
                night_indexes = np.where(
                    np.array([n.split("_")[0] == "night" for n in notes])
                )[0]
                sideways_indexes = np.where(
                    np.array([n.split("_")[1] == "sideways" for n in notes])
                )[0]
            except IndexError:
                raise RuntimeError(
                    "You're using msls_weighted mining but this dataset "
                    + "does not have night/sideways information. Are you using Mapillary SLS?"
                )
            self.weights = np.ones(self.queries_num)
            assert (
                len(night_indexes) != 0 and len(sideways_indexes) != 0
            ), "There should be night and sideways images for msls_weighted mining, but there are none. Are you using Mapillary SLS?"
            self.weights[night_indexes] += self.queries_num / len(night_indexes)
            self.weights[sideways_indexes] += self.queries_num / len(sideways_indexes)
            self.weights /= self.weights.sum()

    def __getitem__(self, index):
        if self.is_inference:
            # At inference time return the single image. This is used for caching or computing NetVLAD's clusters
            return super().__getitem__(index)
        query_index, best_positive_index, neg_indexes = torch.split(
            self.triplets_global_indexes[index], (1, 1, self.neg_num_per_query)
        )
        query = self.train_preprocess(path_to_pil_img(self.queries_paths[query_index]))
        positive = self.train_preprocess(
            path_to_pil_img(self.database_paths[best_positive_index])
        )
        negatives = [
            self.train_preprocess(path_to_pil_img(self.database_paths[i]))
            for i in neg_indexes
        ]
        images = torch.stack((query, positive, *negatives), 0)
        triplets_local_indexes = torch.empty((0, 3), dtype=torch.int)
        for neg_num in range(len(neg_indexes)):
            triplets_local_indexes = torch.cat(
                (
                    triplets_local_indexes,
                    torch.tensor([0, 1, 2 + neg_num]).reshape(1, 3),
                )
            )
        return images, triplets_local_indexes, self.triplets_global_indexes[index]

    def __len__(self):
        if self.is_inference:
            # At inference time return the number of images. This is used for caching or computing NetVLAD's clusters
            return super().__len__()
        else:
            return len(self.triplets_global_indexes)

    def compute_triplets(self, args, model):
        self.is_inference = True
        if self.mining == "full":
            self.compute_triplets_full(args, model)
        elif self.mining == "partial" or self.mining == "msls_weighted":
            self.compute_triplets_partial(args, model)
        elif self.mining == "random":
            self.compute_triplets_random(args, model)

    @staticmethod
    def compute_cache(args, model, subset_ds, cache_shape):
        """Compute the cache containing features of images, which is used to
        find best positive and hardest negatives."""
        subset_dl = DataLoader(
            dataset=subset_ds,
            num_workers=args.num_workers,
            batch_size=args.infer_batch_size,
            shuffle=False,
            pin_memory=(args.device == "cuda"),
        )
        model = model.eval()

        # RAMEfficient2DMatrix can be replaced by np.zeros, but using
        # RAMEfficient2DMatrix is RAM efficient for full database mining.
        cache = RAMEfficient2DMatrix(cache_shape, dtype=np.float32)
        with torch.no_grad():
            progress_bar = tqdm(total=len(subset_dl), desc="Computing Cache")
            for images, indexes in subset_dl:
                images = images.to(args.device)
                features = model(images)
                cache[indexes.numpy()] = features.cpu().numpy()
                progress_bar.update(1)
            progress_bar.close()
        return cache

    def get_query_features(self, query_index, cache):
        query_features = cache[query_index + self.database_num]
        if query_features is None:
            raise RuntimeError(
                f"For query {self.queries_paths[query_index]} "
                + f"with index {query_index} features have not been computed!\n"
                + "There might be some bug with caching"
            )
        return query_features

    def get_best_positive_index(self, args, query_index, cache, query_features):
        positives_features = cache[self.hard_positives_per_query[query_index]]
        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(positives_features)
        # Search the best positive (within 10 meters AND nearest in features space)
        _, best_positive_num = faiss_index.search(query_features.reshape(1, -1), 1)
        best_positive_index = self.hard_positives_per_query[query_index][
            best_positive_num[0]
        ].item()
        return best_positive_index

    def get_hardest_negatives_indexes(self, args, cache, query_features, neg_samples):
        neg_features = cache[neg_samples]
        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(neg_features)
        # Search the 10 nearest negatives (further than 25 meters and nearest in features space)
        _, neg_nums = faiss_index.search(
            query_features.reshape(1, -1), self.neg_num_per_query
        )
        neg_nums = neg_nums.reshape(-1)
        neg_indexes = neg_samples[neg_nums].astype(np.int32)
        return neg_indexes

    def compute_triplets_random(self, args, model):
        self.triplets_global_indexes = []
        # Take 1000 random queries
        sampled_queries_indexes = np.random.choice(
            self.queries_num, args.cache_refresh_rate, replace=False
        )
        # Take all the positives
        positives_indexes = [
            self.hard_positives_per_query[i] for i in sampled_queries_indexes
        ]
        positives_indexes = [
            p for pos in positives_indexes for p in pos
        ]  # Flatten list of lists to a list
        positives_indexes = list(np.unique(positives_indexes))

        # Compute the cache only for queries and their positives, in order to find the best positive
        subset_ds = Subset(
            self, positives_indexes + list(sampled_queries_indexes + self.database_num)
        )
        cache = self.compute_cache(
            args, model, subset_ds, (len(self), args.features_dim)
        )

        # This loop's iterations could be done individually in the __getitem__(). This way is slower but clearer (and yields same results)
        progress_bar = tqdm(sampled_queries_indexes, desc="Mining Radom Features")
        for query_index in sampled_queries_indexes:
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(
                args, query_index, cache, query_features
            )

            # Choose some random database images, from those remove the soft_positives, and then take the first 10 images as neg_indexes
            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.random.choice(
                self.database_num,
                size=self.neg_num_per_query + len(soft_positives),
                replace=False,
            )
            neg_indexes = np.setdiff1d(neg_indexes, soft_positives, assume_unique=True)[
                : self.neg_num_per_query
            ]

            self.triplets_global_indexes.append(
                (query_index, best_positive_index, *neg_indexes)
            )
            progress_bar.update(1)
        progress_bar.close()
        # self.triplets_global_indexes is a tensor of shape [1000, 12]
        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes)

    def compute_triplets_full(self, args, model):
        self.triplets_global_indexes = []
        # Take 1000 random queries
        sampled_queries_indexes = np.random.choice(
            self.queries_num, args.cache_refresh_rate, replace=False
        )
        # Take all database indexes
        database_indexes = list(range(self.database_num))
        #  Compute features for all images and store them in cache
        subset_ds = Subset(
            self, database_indexes + list(sampled_queries_indexes + self.database_num)
        )
        cache = self.compute_cache(
            args, model, subset_ds, (len(self), args.features_dim)
        )

        # This loop's iterations could be done individually in the __getitem__(). This way is slower but clearer (and yields same results)
        progress_bar = tqdm(sampled_queries_indexes, desc="Mining Radom Features")
        for query_index in sampled_queries_indexes:
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(
                args, query_index, cache, query_features
            )
            # Choose 1000 random database images (neg_indexes)
            neg_indexes = np.random.choice(
                self.database_num, self.neg_samples_num, replace=False
            )
            # Remove the eventual soft_positives from neg_indexes
            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.setdiff1d(neg_indexes, soft_positives, assume_unique=True)
            # Concatenate neg_indexes with the previous top 10 negatives (neg_cache)
            neg_indexes = np.unique(
                np.concatenate([self.neg_cache[query_index], neg_indexes])
            )
            # Search the hardest negatives
            neg_indexes = self.get_hardest_negatives_indexes(
                args, cache, query_features, neg_indexes
            )
            # Update nearest negatives in neg_cache
            self.neg_cache[query_index] = neg_indexes
            self.triplets_global_indexes.append(
                (query_index, best_positive_index, *neg_indexes)
            )
            progress_bar.update(1)
        progress_bar.close()
        # self.triplets_global_indexes is a tensor of shape [1000, 12]
        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes)

    def compute_triplets_partial(self, args, model):
        self.triplets_global_indexes = []
        # Take 1000 random queries
        if self.mining == "partial":
            sampled_queries_indexes = np.random.choice(
                self.queries_num, args.cache_refresh_rate, replace=False
            )
        elif (
            self.mining == "msls_weighted"
        ):  # Pick night and sideways queries with higher probability
            sampled_queries_indexes = np.random.choice(
                self.queries_num, args.cache_refresh_rate, replace=False, p=self.weights
            )

        # Sample 1000 random database images for the negatives
        sampled_database_indexes = np.random.choice(
            self.database_num, self.neg_samples_num, replace=False
        )
        # Take all the positives
        positives_indexes = [
            self.hard_positives_per_query[i] for i in sampled_queries_indexes
        ]
        positives_indexes = [p for pos in positives_indexes for p in pos]
        # Merge them into database_indexes and remove duplicates
        database_indexes = list(sampled_database_indexes) + positives_indexes
        database_indexes = list(np.unique(database_indexes))

        subset_ds = Subset(
            self, database_indexes + list(sampled_queries_indexes + self.database_num)
        )
        cache = self.compute_cache(
            args, model, subset_ds, cache_shape=(len(self), args.features_dim)
        )

        # This loop's iterations could be done individually in the __getitem__(). This way is slower but clearer (and yields same results)
        progress_bar = tqdm(sampled_queries_indexes, desc="Mining Radom Features")
        for query_index in sampled_queries_indexes:
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(
                args, query_index, cache, query_features
            )

            # Choose the hardest negatives within sampled_database_indexes, ensuring that there are no positives
            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.setdiff1d(
                sampled_database_indexes, soft_positives, assume_unique=True
            )

            # Take all database images that are negatives and are within the sampled database images (aka database_indexes)
            neg_indexes = self.get_hardest_negatives_indexes(
                args, cache, query_features, neg_indexes
            )
            self.triplets_global_indexes.append(
                (query_index, best_positive_index, *neg_indexes)
            )
            progress_bar.update(1)
        progress_bar.close()
        # self.triplets_global_indexes is a tensor of shape [1000, 12]
        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes)


class RAMEfficient2DMatrix:
    """This class behaves similarly to a numpy.ndarray initialized
    with np.zeros(), but is implemented to save RAM when the rows
    within the 2D array are sparse. In this case it's needed because
    we don't always compute features for each image, just for few of
    them"""

    def __init__(self, shape, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        self.matrix = [None] * shape[0]

    def __setitem__(self, indexes, vals):
        assert vals.shape[1] == self.shape[1], f"{vals.shape[1]} {self.shape[1]}"
        for i, val in zip(indexes, vals):
            self.matrix[i] = val.astype(self.dtype, copy=False)

    def __getitem__(self, index):
        if hasattr(index, "__len__"):
            return np.array([self.matrix[i] for i in index])
        else:
            return self.matrix[index]



################################################### Data Module ###################################################


class TripletDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning Data Module for handling triplet datasets.

    This module prepares the dataset for training and validation, handling the necessary transformations and data loading.

    Parameters:
    - args (Namespace): Arguments containing dataset and model configuration.
    - test_preprocess (function): Transformation function for test data.
    - train_preprocess (function, optional): Transformation function for train data.
    """

    def __init__(self, args, test_preprocess, train_preprocess=None):
        super().__init__()
        self.args = args
        self.test_preprocess = test_preprocess
        if train_preprocess is not None:
            self.train_preprocess = train_preprocess
        else:
            self.train_preprocess = test_preprocess

        self.pin_memory = True if self.args.device == "cuda" else False

    def setup(self, stage=None):
        # Split data into train, validate, and test sets]

        self.train_dataset = TripletsDataset(self.args, self.train_preprocess, self.test_preprocess, split="train")
        self.test_dataset = TripletsDataset(self.args, self.test_preprocess, self.test_preprocess, split="test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.args.train_batch_size, num_workers=self.args.num_workers, pin_memory=self.pin_memory, shuffle=True,
            collate_fn=collate_fn, drop_last=True,
        )
    
    def val_dataloader(self):
        queries_ds = ImageIdxDataset(self.test_dataset.queries_paths, self.test_preprocess)
        queries_dl = DataLoader(queries_ds, num_workers=self.args.num_workers, batch_size=self.args.infer_batch_size, pin_memory=(self.args.device == "cuda"))
        database_ds = ImageIdxDataset(self.test_dataset.database_paths, self.test_preprocess)
        database_dl = DataLoader(database_ds, num_workers=self.args.num_workers, batch_size=self.args.infer_batch_size, pin_memory=(self.args.device == "cuda"))
        return queries_dl, database_dl
    







################################################### Training Module ###################################################


class TripletModule(pl.LightningModule):
    """
    A PyTorch Lightning Module for training with triplet loss.

    This module encapsulates the model training process using triplet loss. It handles the training steps, validation, and testing, as well as optimizer configuration.

    Parameters:
    - args (Namespace): Arguments containing model and training configuration.
    - model (nn.Module): The neural network model to train.
    - datamodule (TripletDataModule): The data module containing training and validation data.

    Methods:
    - features_size(): Returns the size of the features extracted by the model.
    - on_train_start(): Prepares the dataset for training.
    - on_train_epoch_end(): Updates the mining of triplets at the end of each training epoch.
    - forward(x): Forward pass through the model.
    - training_step(batch, batch_idx): Defines the training step.
    - on_validation_epoch_end(outputs): Handles actions at the end of a validation epoch.
    - on_test_epoch_end(batch, batch_idx, dataloader_idx): Handles actions at the end of a test epoch.
    - configure_optimizers(): Configures the model optimizers.
    """

    def __init__(self, args, model, datamodule):
        super().__init__()
        self.args = args
        self.model = model
        self.loss_fn = get_loss_function(args)
        self.save_hyperparameters(ignore=["model"])
        self.datamodule = datamodule
        self.features_dim = self.features_size()
        self.args.features_dim = self.features_dim


    def features_size(self):
        img = Image.open(self.datamodule.train_dataset.queries_paths[0]).convert("RGB")
        img = self.datamodule.test_preprocess(img)
        with torch.no_grad():
            features = self.model(img[None, :].to(self.args.device)).detach().cpu()
            return features.size(1)

    def setup(self, stage):
        self.datamodule.train_dataset.is_inference = True
        self.datamodule.train_dataset.compute_triplets(self.args, self.model)
        self.datamodule.train_dataset.is_inference = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, local_indicies, _ = batch
        
        features = self.model(images)
        loss = 0
        local_indicies = torch.transpose(local_indicies.view(self.args.train_batch_size, self.args.neg_num_per_query, 3), 1, 0)
        for triplets in local_indicies:
            queries_indexes, positives_indexes, negatives_indexes = triplets.T

            loss += self.loss_fn(
                features[queries_indexes],
                features[positives_indexes],
                features[negatives_indexes],
            )

        loss /= self.args.train_batch_size * self.args.neg_num_per_query
        self.log("train_loss", loss)
        return loss
    
    def on_train_epoch_end(self):
        self.datamodule.train_dataset.is_inference = True
        self.datamodule.train_dataset.compute_triplets(self.args, self.model)
        self.datamodule.train_dataset.is_inference = False


    
    def on_validation_epoch_start(self):
        self.validation_queries_descs = np.empty((self.datamodule.test_dataset.queries_num, self.features_dim), dtype=np.float32)
        self.validation_database_descs = np.empty((self.datamodule.test_dataset.database_num, self.features_dim), dtype=np.float32)
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        batch, indicies = batch
        features = self.model(batch)

        if dataloader_idx == 0:
            self.validation_queries_descs[indicies.cpu().numpy(), :] = features.detach().cpu().numpy()
        elif dataloader_idx == 1:
            self.validation_database_descs[indicies.cpu().numpy(), :] = features.detach().cpu().numpy()
            
        

    def on_validation_epoch_end(self):  
        if self.args.loss_distance == "cosine":
            faiss.normalize_L2(self.validation_database_descs)
            index = faiss.IndexFlatIP(self.features_dim)
            index.add(self.validation_database_descs)
            faiss.normalize_L2(self.validation_queries_descs)
        elif self.args.loss_distance == "l2":
            index = faiss.IndexFlatL2(self.features_dim)
            index.add(self.validation_database_descs)

        distances, predictions = index.search(self.validation_queries_descs, max(self.args.recall_values))
        #### For each query, check if the predictions are correct
        positives_per_query = self.datamodule.test_dataset.soft_positives_per_query
        # args.recall_values by default is [1, 5, 10, 20]
        recalls = np.zeros(len(self.args.recall_values))
        for query_index, pred in enumerate(predictions):
            for i, n in enumerate(self.args.recall_values):
                if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                    recalls[i:] += 1
                    break

        # Divide by the number of queries*100, so the recalls are in percentages
        recalls = recalls / self.datamodule.test_dataset.queries_num * 100
        recalls_str = ", ".join([f"R@{val}: {rec:.4f}" for val, rec in zip(self.args.recall_values, recalls)])


        print("")
        print("")
        print(recalls_str)
        print("")
        print("")
        for i, recall in enumerate(recalls):
            self.log("recallat" + str(self.args.recall_values[i]), recall, on_epoch=True)

        return recalls[1]
    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return optimizer

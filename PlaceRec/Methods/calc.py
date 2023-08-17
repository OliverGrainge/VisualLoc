import torch 
import torchvision 
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import cv2
import random
import numpy as np
import torch.nn.functional as F
import os
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from glob import glob
from PIL import Image
from .base_method import BaseTechnique
from typing import Tuple
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import sklearn

PLACES_365_DIRECTORY = "/home/oliver/Documents/github/Datasets/Places/Places365"


package_directory = os.path.dirname(os.path.abspath(__file__))

class Places365(Dataset):
    def __init__(self, root=None, split="train"):
        catagory_paths = glob(root + "/" + split + "/*")
        self.images = []
        for cat in catagory_paths:
            self.images += sorted(glob(cat + '/*'))
        self.images = np.array(self.images)

    def __len__(self):
        return self.images.size

    def __getitem__(self, idx):
        return Image.open(self.images[idx])
        


class calcDataset(Dataset):
    
    def __init__(self, standard_dataset):
        self.ds = standard_dataset

        size = ((160, 120))

        # preprocess with no transformation
        self.preprocess1 = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(size, antialias=True),
            transforms.ToTensor()
        ])
        
        # preprocess with random perspective
        self.preprocess2 = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((int(size[0] * 1.7), int(size[1] * 1.7)), antialias=True),
            transforms.RandomPerspective(distortion_scale=0.4, p = 1.),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ])


        winsize = (64, 64)
        blocksize = (32, 32)
        blockstride = (32, 32)
        cellsize = (16, 16)
        nbins = 9

        self.hog = cv2.HOGDescriptor(winsize, blocksize, blockstride, cellsize, nbins)

    def __len__(self):
        return self.ds.__len__()


    def __getitem__(self, idx):
        img = self.ds.__getitem__(idx)

        # preprocess the pair of images
        imgs = [self.preprocess1(img), self.preprocess2(img)]

        # shuffle them so target and input selection are random
        random.shuffle(imgs)

        target = np.array((imgs[1]*255).type(torch.uint8))[0]
        target = torch.Tensor(self.hog.compute(target))

        return imgs[0], target[None, :]





class calcDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=96):
        super().__init__()
        self.batch_size = batch_size


    def setup(self, stage=None):
        train_ds = Places365(root=PLACES_365_DIRECTORY, split="train")
        self.train_dataset = calcDataset(train_ds)
        test_ds = Places365(root=PLACES_365_DIRECTORY, split="val")
        self.val_dataset = calcDataset(test_ds)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16)


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)


    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)






class CALCModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.input_dim = (1, 160, 120)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(5,5), stride=2, padding=4)
        self.norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(4,4), stride=1, padding=2)
        self.norm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 4, kernel_size=(3,3), stride=1, padding=0)
        self.fc1 = nn.Linear(936, 1064)
        self.fc2 = nn.Linear(1064, 2048)
        self.fc3 = nn.Linear(2048, 4032)

        self.pool = nn.MaxPool2d(kernel_size=(3,3), stride=2)



    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.norm1(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.norm2(x)

        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

    
    def describe(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.norm1(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.norm2(x)

        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        return x


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat[:, None, :], y)
        self.log("train_loss", loss)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat[:, None, :], y)
        self.log("val_loss", loss)
        #return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=9e-4, momentum=0.9, weight_decay=5e-4)




class CALC(BaseTechnique):
    
    def __init__(self):
        super().__init__()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = CALCModule.load_from_checkpoint(package_directory + "/weights/calc_weights.ckpt").to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Resize((160, 120), antialias=True)
        ])


        self.map = None
        self.map_desc = None
        self.query_desc = None
        self.name = "calc"




    def compute_query_desc(self, images: torch.Tensor = None, dataloader: torch.utils.data.dataloader.DataLoader=None, pbar: bool=True) -> dict:
        if images is not None and dataloader is None:
            all_desc = self.model.describe(images.to(self.device)).detach().cpu().numpy()
        elif dataloader is not None and images is None:
            all_desc = []
            for batch in tqdm(dataloader, desc="Computing CALC Query Desc", disable=not pbar):
                all_desc.append(self.model.describe(batch.to(self.device)).detach().cpu().numpy())
            all_desc = np.vstack(all_desc)
        
        
        query_desc = {"query_descriptors": all_desc}
        self.set_query(query_desc)
        return query_desc


    def compute_map_desc(self, images: torch.Tensor = None, dataloader: torch.utils.data.dataloader.DataLoader=None, pbar: bool=True) -> dict:
        
        if images is not None and dataloader is None:
            all_desc = self.model.describe(images.to(self.device)).detach().cpu().numpy()
        elif dataloader is not None and images is None:
            all_desc = []
            for batch in tqdm(dataloader, desc="Computing CALC Map Desc", disable=not pbar):
                all_desc.append(self.model.describe(batch.to(self.device)).detach().cpu().numpy())
            all_desc = np.vstack(all_desc)
        else: 
            raise Exception("can only pass 'images' or 'dataloader'")

        map_desc = {'map_descriptors': all_desc}
        self.set_map(map_desc)

        return map_desc


    def set_query(self, query_descriptors: dict) -> None:
        self.query_desc = query_descriptors


    def set_map(self, map_descriptors: dict) -> None:
        self.map_desc = map_descriptors
        try: 
            # try to implement with faiss
            self.map = faiss.IndexFlatIP(map_descriptors["map_descriptors"].shape[1])
            faiss.normalize_L2(map_descriptors["map_descriptors"])
            self.map.add(map_descriptors["map_descriptors"])

        except: 
            # faiss is not available on unix or windows systems. In this case 
            # implement with scikit-learn
            self.map = NearestNeighbors(n_neighbors=10, algorithm='auto', 
                                        metric='cosine').fit(map_descriptors["map_descriptors"])


    def place_recognise(self, images: torch.Tensor=None, dataloader: torch.utils.data.dataloader.DataLoader=None, top_n: int=1, pbar: bool=True) -> Tuple[np.ndarray, np.ndarray]:
        desc = self.compute_query_desc(images=images, dataloader=dataloader, pbar=pbar)
        if isinstance(self.map, sklearn.neighbors._unsupervised.NearestNeighbors):
            dist, idx = self.map.kneighbors(desc["query_descriptors"])
            return idx[:, :top_n], 1 - dist[:, :top_n]
        else: 
            faiss.normalize_L2(desc["query_descriptors"])
            dist, idx = self.map.search(desc["query_descriptors"], top_n)
            return idx, dist


    def similarity_matrix(self, query_descriptors: dict, map_descriptors: dict) -> np.ndarray:
        if self.device == 'cuda': 
            return cosine_similarity_cuda(map_descriptors["map_descriptors"], 
                                          query_descriptors["query_descriptors"]).astype(np.float32)
        else: 
            return cosine_similarity(map_descriptors["map_descriptors"],
                                    query_descriptors["query_descriptors"]).astype(np.float32)


    
    def save_descriptors(self, dataset_name: str) -> None:
        if not os.path.isdir(package_directory + "/descriptors/" + dataset_name):
            os.makedirs(package_directory + "/descriptors/" + dataset_name)
        with open(package_directory + "/descriptors/" + dataset_name + "/" + self.name + "_query.pkl", "wb") as f:
            pickle.dump(self.query_desc, f)
        with open(package_directory + "/descriptors/" + dataset_name + "/" + self.name + "_map.pkl", "wb") as f:
            pickle.dump(self.map_desc, f)
        

    def load_descriptors(self, dataset_name: str) -> None:
        if not os.path.isdir(package_directory + "/descriptors/" + dataset_name):
            raise Exception("Descriptor not yet computed for: " + dataset_name)
        with open(package_directory + "/descriptors/" + dataset_name + "/" + self.name + "_query.pkl", "rb") as f:
            self.query_desc = pickle.load(f)
        with open(package_directory + "/descriptors/" + dataset_name + "/" + self.name + "_map.pkl", "rb") as f:
            self.map_desc = pickle.load(f)
            self.set_map(self.map_desc)



if __name__ == '__main__': 
    torch.set_float32_matmul_precision('medium')
    model = CALC()
    model.train()
    data = calcDataModule()
    logger = TensorBoardLogger("tb_logs", name="calc")
    checkpoint_callback = ModelCheckpoint(dirpath="weights/",
                                          filename="calc_weights",
                                          monitor="val_loss", save_top_k=1,
                                          mode='min')

    trainer = pl.Trainer(accelerator="gpu", devices=1, 
                         logger=logger, max_epochs=42, callbacks=[checkpoint_callback])

    trainer.fit(model=model, datamodule=data)
    trainer.test(model=model, datamodule=datamodule)

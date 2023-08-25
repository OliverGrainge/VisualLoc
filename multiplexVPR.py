from PlaceRec.Methods.base_method import BaseFunctionality, BaseTechnique
import torch
from PlaceRec.utils import get_method, ImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors
import os
from sklearn.metrics.pairwise import cosine_similarity
from train_selection import ResNet18, select_transform
from typing import Tuple
import pickle
from torchvision import transforms
from PIL import Image
import faiss

METHODS = ["netvlad", "hybridnet", "amosnet"]
SELECTION_MODEL_PATH = "/home/oliver/Documents/github/VisualLoc/SelectionNetworkCheckpoints/combined_largesets_recall@1_resnet18-epoch=45-val_loss=0.52.ckpt"
BATCHSIZE = 160


package_directory = os.path.dirname(os.path.abspath(__file__))


class MultiPlexVPR(BaseFunctionality):
    def __init__(
        self,
        selection_weights="/home/oliver/Documents/github/VisualLoc/SelectionNetworkCheckpoints/combined_largesets_recall@1_resnet18-epoch=45-val_loss=0.52.ckpt",
        methods=METHODS,
    ):
        super().__init__()
        self.name = "multiplexvpr"

        # instantiate vpr techniques
        self.methods = [get_method(method_name) for method_name in methods]
        for method in self.methods:
            method.device = "cpu"
            method.model.to("cpu")

        # instantiate selection model
        self.selection_model = ResNet18.load_from_checkpoint(
            selection_weights, output_dim=len(methods)
        ).to(self.device)
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def select_network_preprocess(self, img1, img2):
        return torch.vstack((img1, img2))[None, :]

    def compute_selection(self, query_img, ref_img):
        if ref_img is None:
            return 0
        input = self.select_network_preprocess(query_img, ref_img)
        logits = self.selection_model(input.to(self.device)).detach().cpu()
        print("=====", logits.numpy())
        selection = logits.argmax().item()
        return selection

    def compute_query_desc(
        self,
        query_images: torch.Tensor = None,
        map_images: torch.Tensor = None,
        query_dataloader: torch.utils.data.dataloader.DataLoader = None,
        map_dataloader: torch.utils.data.dataloader.DataLoader = None,
        pbar: bool = True,
    ) -> dict:
        assert map_dataloader is not None
        assert query_dataloader is not None

        # brute force apply all techniques beforehand. This makes things simpler.
        q_img_paths = query_dataloader.dataset.img_paths
        m_img_paths = map_dataloader.dataset.img_paths
        for method in self.methods:
            ds_q = ImageDataset(img_paths=q_img_paths, preprocess=method.preprocess)
            dl_q = DataLoader(ds_q, shuffle=False, batch_size=BATCHSIZE)
            ds_m = ImageDataset(img_paths=m_img_paths, preprocess=method.preprocess)
            dl_m = DataLoader(ds_m, shuffle=False, batch_size=BATCHSIZE)
            q_desc = method.compute_query_desc(dataloader=dl_q)
            m_desc = method.compute_map_desc(dataloader=dl_m)
            method.set_query(q_desc)
            method.set_map(m_desc)

        all_desc = []
        method_selections = []
        ref_img = None
        count = 0
        for batch in tqdm(
            query_dataloader, desc="Computing MultisplexVPR Desc", disable=not pbar
        ):
            for query_image in batch:
                # 1. make a technique selection.
                select_idx = self.compute_selection(query_image, ref_img)

                # 2. Compute the descriptors computed by the selected method
                query_desc = (
                    self.methods[select_idx]
                    .query_desc["query_descriptors"][count]
                    .flatten()
                )

                # 3 Append the descriptors and method selection to a list
                all_desc.append(query_desc)
                method_selections.append(select_idx)

                # 4. retrieve the reference image.
                idx, score = self.methods[select_idx].place_recognise(
                    descriptors=query_desc[None, :], top_n=1
                )
                ref_path = map_dataloader.dataset.img_paths[idx.flatten().item()]
                ref_img = self.preprocess(
                    Image.fromarray(np.array(Image.open(ref_path))[:, :, :3])
                )
                count += 1

        all_desc = {
            "query_descriptors": all_desc,
            "selections": np.array(method_selections),
        }
        self.set_query(all_desc)
        return all_desc

    def compute_map_desc(
        self,
        images: torch.Tensor = None,
        dataloader: torch.utils.data.dataloader.DataLoader = None,
        pbar: bool = True,
    ) -> dict:
        if images is not None and dataloader is not None:
            raise NotImplementedError
        elif dataloader is not None and images is None:
            img_paths = dataloader.dataset.img_paths
            progress_bar = tqdm(
                total=dataloader.dataset.__len__() * len(METHODS),
                desc="Computing MultiPlexVPR Map Descriptors",
                leave=True,
                disable=not pbar,
            )
            all_desc = []
            for method in self.methods:
                ds = ImageDataset(img_paths=img_paths, preprocess=method.preprocess)
                dl = DataLoader(ds, shuffle=False, batch_size=BATCHSIZE)
                all_desc.append(
                    method.compute_map_desc(dataloader=dl, pbar=False)[
                        "map_descriptors"
                    ]
                )
                progress_bar.update(ds.__len__())
            map_desc = {"map_descriptors": all_desc}
            self.set_map(map_desc)
            return map_desc

    def set_map(self, map_descriptors: dict) -> None:
        self.map_desc = map_descriptors
        all_desc = map_descriptors["map_descriptors"]
        self.map = []
        for desc in all_desc:
            self.map_desc = map_descriptors
            try:
                map_idx = faiss.IndexFlatIP(desc.shape[1])
                faiss.normalize_L2(desc)
                map_idx.add(desc)
                self.map.append(map_idx)

            except:
                map_idx = NearestNeighbors(
                    n_neighbors=10, algorithm="auto", metric="cosine"
                ).fit(desc)
                self.map.append(map_idx)

    def similarity_matrix(
        self, query_descriptors: np.ndarray, map_descriptors: np.ndarray
    ) -> np.ndarray:
        q_desc = query_descriptors["query_descriptors"]
        selections = query_descriptors["selections"]
        masks = [np.where(np.array(selections) == i) for i in range(len(METHODS))]
        select_descriptors = [np.array(q_desc, dtype=object)[mask] for mask in masks]
        select_descriptors = [
            np.vstack(d) if len(d) > 0 else np.array([]) for d in select_descriptors
        ]
        m_desc = map_descriptors["map_descriptors"]
        sep_S = [
            cosine_similarity(d, m_desc[i])
            if len(d) != 0
            else np.zeros(m_desc[0].shape[0])
            for i, d in enumerate(select_descriptors)
        ]
        S = np.zeros((len(selections), m_desc[0].shape[0]))
        for i, mask in enumerate(masks):
            S[mask] = sep_S[i]
        S = S.transpose()
        return S

    def place_recognise(
        self,
        images: torch.Tensor = None,
        dataloader: torch.utils.data.dataloader.DataLoader = None,
        pbar: bool = True,
        top_n: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

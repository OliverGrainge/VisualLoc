import torch
from PlaceRec.utils import get_dataset, ImageIdxDataset
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import faiss
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image
import os

DATASET_NAME = "Pitts30k"
NUM_WORKERS = 0
BATCH_SIZE = 32
DEVICE = "cpu"


valid_transform = T.Compose(
    [
        T.Resize((320, 320), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

model = torch.load("", map_location=DEVICE)
model_pruned = torch.load("", map_location=DEVICE)

dataset = get_dataset(DATASET_NAME)

query_paths = dataset.query_paths
map_paths = dataset.map_paths

query_dataset = ImageIdxDataset(query_paths, preprocess=valid_transform)
map_dataset = ImageIdxDataset(map_paths, preprocess=valid_transform)

query_loader = DataLoader(query_dataset, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)
map_loader = DataLoader(map_dataset, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)


def compute_descriptors(model, loader):
    model.eval()
    desc = []
    for batch in tqdm(loader):
        batch = batch.to(DEVICE)
        features = model(batch).detach().cpu().numpy()
        desc.append(features)
    return np.vstack(desc).astype(np.float32)


def plot_image_grid(image_paths, title, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    fig.suptitle(title, fontsize=16)

    for ax, img_path in zip(axes.flat, image_paths):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis("off")

    # Hide any remaining empty subplots
    for i in range(len(image_paths), rows * cols):
        fig.delaxes(axes.flat[i])

    plt.show()


query_desc = compute_descriptors(model, query_loader)
map_desc = compute_descriptors(model, map_loader)
query_desc_pruned = compute_descriptors(model_pruned, query_loader)
map_desc_pruned = compute_descriptors(model_pruned, map_loader)

faiss.normalize_L2(query_desc)
faiss.normalize_L2(map_desc)
faiss.normalize_L2(query_desc_pruned)
faiss.normalize_L2(map_desc_pruned)

map = faiss.IndexFlatIP(map_desc)
map_pruned = faiss.IndexFlatIP(map_desc_pruned)

matches = map.search(query_desc, map_desc.shape[0])[0]
matches_pruned = map_pruned.search(query_desc_pruned, map_desc_pruned.shape[0])[0]

gt = dataset.gt

result = [1 if set(p).intersection(set(gt)) else 0 for p, gt in zip(matches, gt)]
result_pruned = [
    1 if set(p).intersection(set(gt)) else 0 for p, gt in zip(matches_pruned, gt)
]

TP_indices = np.array(
    [i for i in range(len(result)) if result[i] == 1 and result_pruned[i] == 1]
)
FP_indices = np.array(
    [i for i in range(len(result)) if result[i] == 0 and result_pruned[i] == 1]
)
FN_indices = np.array(
    [i for i in range(len(result)) if result[i] == 1 and result_pruned[i] == 0]
)
TN_indices = np.array(
    [i for i in range(len(result)) if result[i] == 0 and result_pruned[i] == 0]
)


TP_paths = dataset.query_paths[TP_indices]
FP_paths = dataset.query_paths[FP_indices]
FN_paths = dataset.query_paths[FN_indices]
TN_paths = dataset.query_paths[TN_indices]


rows, cols = 4, 4  # Adjust these values as needed

# Ensure the paths arrays are numpy arrays
TP_paths = np.array(TP_paths)
FP_paths = np.array(FP_paths)
FN_paths = np.array(FN_paths)
TN_paths = np.array(TN_paths)

# Plot grids
plot_image_grid(TP_paths[: rows * cols], "True Positives (TP)", rows, cols)
plot_image_grid(FP_paths[: rows * cols], "False Positives (FP)", rows, cols)
plot_image_grid(FN_paths[: rows * cols], "False Negatives (FN)", rows, cols)
plot_image_grid(TN_paths[: rows * cols], "True Negatives (TN)", rows, cols)

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
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import norm
from typing import Union
import os

DOMAIN = "embedding_confusion"  # Image embedding_confusion or embedding_comparison
THRESHOLD = 0.7
DATASET_NAME = "InriaHolidays"
NUM_WORKERS = 0
BATCH_SIZE = 32
DEVICE = "mps"
MODEL_PATH = "Checkpoints/ResNet34_ConvAP_agg_0.00_sparsity_0.038_R1_0.917.ckpt"
MODEL_PATH_PRUNED = "Checkpoints/ResNet34_ConvAP_agg_0.75_sparsity_0.435_R1_0.897.ckpt"


def thresholding(S: np.ndarray, thresh: Union[str, float]) -> np.ndarray:
    """
    Applies thresholding on a similarity matrix S based on a given threshold or
    an automatic thresholding method.

    The automatic thresholding is based on Eq. 2-4 in Schubert, S., Neubert, P. &
    Protzel, P. (2021). Beyond ANN: Exploiting Structural Knowledge for Efficient
    Place Recognition. In Proc. of Intl. Conf. on Robotics and Automation (ICRA).
    DOI: 10.1109/ICRA48506.2021.9561006

    Args:
        S (np.ndarray): A two-dimensional similarity matrix with continuous values.
            Higher values indicate higher similarity.
        thresh (Union[str, float]): A threshold value or the string 'auto' to apply
            automatic thresholding.

    Returns:
        np.ndarray: A two-dimensional boolean matrix with the same shape as S,
            where values greater or equal to the threshold are marked as True.
    """
    if thresh == "auto":
        mu = np.median(S)
        sig = np.median(np.abs(S - mu)) / 0.675
        thresh = norm.ppf(1 - 1e-6, loc=mu, scale=sig)

    M = S >= thresh

    return M


def createPR(S_in, GThard, GTsoft=None, matching="multi", n_thresh=100):
    """
    Calculates the precision and recall at n_thresh equally spaced threshold values
    for a given similarity matrix S_in and ground truth matrices GThard and GTsoft for
    single-best-match VPR or multi-match VPR.

    The matrices S_in, GThard and GTsoft are two-dimensional and should all have the
    same shape.
    The matrices GThard and GTsoft should be binary matrices, where the entries are
    only zeros or ones.
    The matrix S_in should have continuous values between -Inf and Inf. Higher values
    indicate higher similarity.
    The string matching should be set to either "single" or "multi" for single-best-
    match VPR or multi-match VPR.
    The integer n_tresh controls the number of threshold values and should be >1.
    """

    assert (
        S_in.shape == GThard.shape
    ), "S_in, GThard and GTsoft must have the same shape"
    assert S_in.ndim == 2, "S_in, GThard and GTsoft must be two-dimensional"
    assert matching in [
        "single",
        "multi",
    ], "matching should contain one of the following strings: [single, multi]"
    assert n_thresh > 1, "n_thresh must be >1"

    # ensure logical datatype in GT and GTsoft
    GT = GThard.astype("bool")
    if GTsoft is not None:
        GTsoft = GTsoft.astype("bool")

    # copy S and set elements that are only true in GTsoft to min(S) to ignore them during evaluation
    S = S_in.copy()
    if GTsoft is not None:
        S[GTsoft & ~GT] = S.min()

    # single-best-match or multi-match VPR
    if matching == "single":
        # count the number of ground-truth positives (GTP)
        GTP = np.count_nonzero(GT.any(0))

        # GT-values for best match per query (i.e., per column)
        GT = GT[np.argmax(S, axis=0), np.arange(GT.shape[1])]

        # similarities for best match per query (i.e., per column)
        S = np.max(S, axis=0)

    elif matching == "multi":
        # count the number of ground-truth positives (GTP)
        GTP = np.count_nonzero(GT)  # ground truth positives

    # init precision and recall vectors
    R = [
        0,
    ]
    P = [
        1,
    ]

    # select start and end treshold
    startV = S.max()  # start-value for treshold
    endV = S.min()  # end-value for treshold

    # iterate over different thresholds
    for i in np.linspace(startV, endV, n_thresh):
        B = S >= i  # apply threshold

        TP = np.count_nonzero(GT & B)  # true positives
        FP = np.count_nonzero((~GT) & B)  # false positives

        P.append(TP / (TP + FP))  # precision
        R.append(TP / GTP)  # recall

    return P, R


valid_transform = T.Compose(
    [
        T.Resize((320, 320), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

model = torch.load(MODEL_PATH, map_location=DEVICE)
model_pruned = torch.load(MODEL_PATH_PRUNED, map_location=DEVICE)

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
        imgs = batch[1].to(DEVICE)
        features = model(imgs).detach().cpu().numpy()
        desc.append(features)
    return np.vstack(desc).astype(np.float32)


def plot_image_grid(image_paths, title, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(5, 5))
    fig.suptitle(title, fontsize=16)

    for ax, img_path in zip(axes.flat, image_paths):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis("off")

    # Hide any remaining empty subplots
    for i in range(len(image_paths), rows * cols):
        fig.delaxes(axes.flat[i])

    plt.show(block=False)


query_desc = compute_descriptors(model, query_loader)
map_desc = compute_descriptors(model, map_loader)
query_desc_pruned = compute_descriptors(model_pruned, query_loader)
map_desc_pruned = compute_descriptors(model_pruned, map_loader)

similarity = cosine_similarity(query_desc, map_desc)
similarity_pruned = cosine_similarity(query_desc_pruned, map_desc_pruned)


# predictions = thresholding(similarity, thresh="auto")
# predictions_pruned = thresholding(similarity, thresh="auto")
predictions = (similarity >= THRESHOLD).astype(int)
predictions_pruned = (similarity_pruned >= THRESHOLD).astype(int)


ground_truth_matrix = np.zeros_like(similarity)
gt = dataset.ground_truth()
for query_idx, map_indices in enumerate(gt):
    ground_truth_matrix[query_idx, map_indices] = 1

print(ground_truth_matrix.shape, similarity.shape)

P, R = createPR(similarity, ground_truth_matrix)
plt.plot(R, P, label="Dense")
P, R = createPR(similarity_pruned, ground_truth_matrix)
plt.plot(R, P, label="Pruned")
plt.legend()
plt.show()


cm = confusion_matrix(ground_truth_matrix.flatten(), predictions.flatten())

cm_pruned = confusion_matrix(
    ground_truth_matrix.flatten(), predictions_pruned.flatten()
)


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    plt.figure(figsize=(8, 6))
    # Create the heatmap and capture the Axes object returned by sns.heatmap
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()

    # Add the colorbar correctly by referencing the heatmap (ax)
    plt.colorbar(
        ax.collections[0]
    )  # This uses the mappable object created by sns.heatmap
    plt.show()


plot_confusion_matrix(
    cm, classes=["Non-match", "Match"], normalize=False, title="My Confusion Matrix"
)


faiss.normalize_L2(query_desc)
faiss.normalize_L2(map_desc)
faiss.normalize_L2(query_desc_pruned)
faiss.normalize_L2(map_desc_pruned)

map = faiss.IndexFlatIP(map_desc.shape[1])
map_pruned = faiss.IndexFlatIP(map_desc_pruned.shape[1])
map.add(map_desc)
map_pruned.add(map_desc_pruned)

matches = map.search(query_desc, 1)[1]
matches_pruned = map_pruned.search(query_desc_pruned, 1)[1]
# gt is a list of arrays of containint indicies of the positive matches
gt = dataset.ground_truth()


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

print(len(TP_indices), len(FP_indices), len(FN_indices), len(TN_indices))

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

# Plot gridss
if len(TP_paths) > rows * cols:
    plot_image_grid(TP_paths[: rows * cols], "True Positives (TP)", rows, cols)
if len(FP_paths) > rows * cols:
    plot_image_grid(FP_paths[: rows * cols], "False Positives (FP)", rows, cols)
if len(FN_paths) > rows * cols:
    plot_image_grid(FN_paths[: rows * cols], "False Negatives (FN)", rows, cols)
if len(TN_paths) > rows * cols:
    plot_image_grid(TN_paths[: rows * cols], "True Negatives (TN)", rows, cols)

plt.show(block=True)

labels = np.array(["TP" for _ in range(len(dataset.query_paths))])
labels[FP_indices] = "FP"
labels[FN_indices] = "FN"
labels[TN_indices] = "TN"

descriptors = query_desc_pruned

tsne = TSNE(n_components=2, random_state=42)
descriptors_reduced = tsne.fit_transform(descriptors)

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=descriptors_reduced[:, 0],
    y=descriptors_reduced[:, 1],
    hue=labels,
    palette=sns.color_palette("hsv", len(np.unique(labels))),
    alpha=0.2,
    legend="full",
    s=50,
)
plt.title("Confusion Matrix Embedding results")
plt.show()


labels = ["Dense" for _ in range(len(dataset.query_paths))] + [
    "Pruned" for _ in range(len(dataset.query_paths))
]
inter_size = (
    min(query_desc_pruned.shape[1], query_desc.shape[1], len(dataset.query_paths)) - 1
)
scaler = PCA(n_components=inter_size, random_state=42)
descriptors_pruned_reduced = scaler.fit_transform(query_desc_pruned)
scaler = PCA(n_components=inter_size, random_state=42)
descriptors_reduced = scaler.fit_transform(query_desc)

descriptors_inter = np.vstack([descriptors_reduced, descriptors_pruned_reduced])
scaler = TSNE(n_components=2, random_state=42)
descriptors_reduced = scaler.fit_transform(descriptors_inter)

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=descriptors_reduced[:, 0],
    y=descriptors_reduced[:, 1],
    hue=labels,
    palette=sns.color_palette("hsv", len(np.unique(labels))),
    alpha=0.2,
    legend="full",
    s=50,
)
plt.title("Dense V Pruned Embeddings")
plt.show()

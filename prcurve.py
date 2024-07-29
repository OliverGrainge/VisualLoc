import torch
import os
from PlaceRec.utils import get_dataset
from tqdm import tqdm
import numpy as np
import torchvision.transforms as T
from PlaceRec.utils import ImageIdxDataset
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

DEVICE = "mps"
METHOD = "ConvAP"
AGGREAGATION_RATE = "0.75"
DATASET_NAME = "InriaHolidays"
NUM_WORKERS = 0
BATCH_SIZE = 32

valid_transform = T.Compose(
    [
        T.Resize((320, 320), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_model_paths():
    pths = os.listdir("Checkpoints/")
    filtered_pths = []
    for pth in pths:
        if METHOD in pth and "_" + AGGREAGATION_RATE + "_" in pth and ".ckpt" in pth:
            filtered_pths.append(pth)
    return filtered_pths


def extract_sparsity(path):
    parts = path.split("_")
    num = float(parts[5])
    return num


def compute_descriptors(model, loader):
    model.eval()
    desc = []
    for batch in tqdm(loader):
        imgs = batch[1].to(DEVICE)
        features = model(imgs).detach().cpu().numpy()
        desc.append(features)
    return np.vstack(desc).astype(np.float32)


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


dataset = get_dataset(DATASET_NAME)
gt = dataset.ground_truth()


query_ds = ImageIdxDataset(dataset.query_paths, valid_transform)
map_ds = ImageIdxDataset(dataset.map_paths, valid_transform)
query_loader = DataLoader(query_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
map_loader = DataLoader(map_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

for pth in get_model_paths()[::4]:
    sparsity = extract_sparsity(pth)
    model = torch.load(os.path.join("Checkpoints/", pth), map_location=DEVICE)
    query_desc = compute_descriptors(model, query_loader)
    map_desc = compute_descriptors(model, map_loader)
    similarity = cosine_similarity(query_desc, map_desc)
    ground_truth_matrix = np.zeros_like(similarity)
    for query_idx, map_indices in enumerate(gt):
        ground_truth_matrix[query_idx, map_indices] = 1
    P, R = createPR(similarity, ground_truth_matrix)
    plt.plot(R, P, label=str(sparsity))

plt.legend()
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precission Recall Curve")
plt.show()

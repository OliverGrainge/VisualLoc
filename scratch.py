from PlaceRec.Methods import DenseVLAD
from PlaceRec.utils import get_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from itertools import combinations
import cv2
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(
    "/home/oliver/Documents/github/VisualLoc/SelectionData/nordlands_combined_prevframe_recall@1_train.csv"
)


METHODS = ["hog", "hybridnet", "amosnet", "netvlad", "calc"]
VALUE = "entropy_var"


def calculate_entropy(image):
    """Calculate the entropy of an image."""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist /= hist.sum()
    entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))
    return entropy


def local_entropy(image, ksize=5):
    """Calculate local entropy map using a square window of side length 'ksize'."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the kernel
    kernel = np.ones((ksize, ksize), dtype=np.float32) / (ksize**2)

    # Local sum (convolution with kernel)
    local_sum = cv2.filter2D(gray, cv2.CV_32F, kernel)

    # Local sum of squares (convolution with squared kernel)
    local_sum_sq = cv2.filter2D(gray**2, cv2.CV_32F, kernel)

    # Compute variance using: VAR = E(X^2) - E(X)^2
    local_var = local_sum_sq - local_sum**2

    # Compute standard deviation as sqrt(VAR)
    local_stddev = np.sqrt(np.abs(local_var))

    # Compute local entropy as log2 of standard deviation
    local_entropy = np.log2(local_stddev + np.finfo(float).eps)

    return local_entropy


def compute_ent_stats(df):
    ent_sum = []
    ent_mean = []
    ent_std = []
    ent_var = []

    for img in tqdm(df["query_images"]):
        img = cv2.imread(img)
        ent = local_entropy(img)
        ent_sum.append(ent.sum())
        ent_mean.append(ent.mean())
        ent_std.append(ent.std())
        ent_var.append(ent.var())
    ent_stats = {
        "entropy_sum": ent_sum,
        "entropy_mean": ent_mean,
        "entropy_std": ent_std,
        "entropy_var": ent_var,
    }

    df_new = df.assign(**ent_stats)
    return df_new


# Since the previous code execution state was reset, let's try running the code again.


def compute_average_recallat1(df, method="hybridnet", value="entropy_mean"):
    # Define the entropy ranges
    # Generating ranges automatically with a given step size

    start = 0  # Start of the range
    end = df[value].max()  # End of the range
    step = end / 20  # Step size (fineness of the range)
    ranges = [(i, i + step) for i in np.arange(start, end, step)]

    # Initialize an empty DataFrame to store the results
    result_df = pd.DataFrame(columns=[value, method])

    # Group rows by entropy range and compute the mean value
    for lower, upper in ranges:
        mask = (df[value] >= lower) & (df[method] < upper)
        mean_value = df.loc[mask, method].mean()

        # Append the result to the result DataFrame
        new_row = pd.DataFrame(
            {"entropy_range": [f"{lower}-{upper}"], "mean_value": [mean_value]}
        )
        result_df = pd.concat([result_df, new_row], ignore_index=True)

    # Plotting
    return result_df["entropy_range"].to_numpy(), result_df["mean_value"].to_numpy()
    # plt.figure(figsize=(10, 6))
    # plt.plot(result_df["entropy_range"], result_df["mean_value"], color="blue")
    # plt.xlabel("Entropy Range")
    # plt.ylabel("Mean Value")
    # plt.title("Mean Value by Entropy Range")
    # plt.show()


df = compute_ent_stats(df)


for method in METHODS:
    x, y = compute_average_recallat1(df, method=method, value=VALUE)
    plt.plot(x, y, label=method)

ax = plt.gca()
ax.set_xticklabels([])
plt.xlabel(VALUE)
plt.ylabel("recall@1")
plt.title("Image " + VALUE + " Vs Recall@1")
plt.legend()
plt.show()

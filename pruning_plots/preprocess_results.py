import pandas as pd
import numpy as np


df = pd.read_csv("../results.csv")

# Function to extract sparsity value
def extract_sparsity(path):
    parts = path.split("_")
    num = float(parts[5])
    return num


# Apply the function to the weight_path column
df["sparsity"] = df["weight_path"].apply(extract_sparsity)

# Convert the extracted values to numeric type
df["sparsity"] = pd.to_numeric(df["sparsity"])
df["agg_rate"] = df["weight_path"].str.extract("agg_([0-9]+\.?[0-9]*)")
df["agg_rate"] = pd.to_numeric(df["agg_rate"])

df.to_csv("results.csv")

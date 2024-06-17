import pandas as pd
import numpy as np


df = pd.read_csv("../results.csv")

print(df["weight_path"])
df["agg_rate"] = df["weight_path"].str.extract("agg_([0-9]+\.?[0-9]*)")
df["agg_rate"] = pd.to_numeric(df["agg_rate"])

df.to_csv("results.csv")

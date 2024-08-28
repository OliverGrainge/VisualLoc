import pandas as pd
import numpy as np

# Load data
df_acc = pd.read_csv("accuracy_results.csv")
df_lat = pd.read_csv("latency_results.csv")

# Function to extract sparsity value
def extract_sparsity(path):
    parts = path.split("_")
    num = float(parts[5])
    return num


# Apply the function to the weight_path column
df_acc["sparsity"] = df_acc["weight_path"].apply(extract_sparsity)

# Convert the extracted values to numeric type
df_acc["sparsity"] = pd.to_numeric(df_acc["sparsity"])
df_acc["agg_rate"] = df_acc["weight_path"].str.extract("agg_([0-9]+\.?[0-9]*)")
df_acc["agg_rate"] = pd.to_numeric(df_acc["agg_rate"])

# Identify columns in df_lat with 'lat' in the name
lat_columns = [col for col in df_lat.columns if "lat" in col]

# Ensure df_acc is ready to receive the data by checking if it has an index set, if not, you might set it or ensure both DataFrames align by default
# This part might need modification based on how you want to align the data:
# For example, if both DataFrames should align by a specific column, use:
# df_acc.set_index('some_column', inplace=True)
# df_lat.set_index('some_column', inplace=True)

# Update df_acc with latency data
df_acc.update(df_lat[lat_columns])

# Save the updated DataFrame
df_acc.to_csv("results.csv")
print(df_acc)

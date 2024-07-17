import pandas as pd
import numpy as np

# Load the datasets
df = pd.read_csv("../accuracy_results.csv")
df_lat = pd.read_csv("../latency_results.csv")

# Set the index for the latency DataFrame
df_lat.set_index("weight_path", inplace=True)

# Define the columns to be populated with latency data
cols = [
    "extraction_lat_cpu_bs1",
    "extraction_lat_gpu_bs1",
    "Pitts30k_Val_matching_lat",
    "Pitts30k_Val_total_cpu_lat_bs1",
    "Pitts30k_Val_total_gpu_lat_bs1",
    "AmsterTime_matching_lat",
    "AmsterTime_total_cpu_lat_bs1",
    "AmsterTime_total_gpu_lat_bs1",
    "extraction_lat_cpu_bs1",
    "extraction_lat_gpu_bs1",
]

# Initialize the columns in the accuracy DataFrame
for col in cols:
    df[col] = None

# Function to extract sparsity value from weight_path
def extract_sparsity(path):
    parts = path.split("_")
    num = float(parts[5])
    return num


# Apply the function to extract sparsity
df["sparsity"] = df["weight_path"].apply(extract_sparsity)

# Convert the extracted values to numeric type
df["sparsity"] = pd.to_numeric(df["sparsity"])
df["agg_rate"] = df["weight_path"].str.extract("agg_([0-9]+\.?[0-9]*)")
df["agg_rate"] = pd.to_numeric(df["agg_rate"])

# Set the index for the accuracy DataFrame
df.set_index("weight_path", inplace=True)

# Populate the latency columns in the accuracy DataFrame
for index, row in df_lat.iterrows():
    for col in cols:
        df.loc[index, col] = row[col]

# Multiply every column that contains "R1" by 100
r1_columns = [col for col in df.columns if "R1" in col]
df[r1_columns] = df[r1_columns] * 100

# Print a sample column to verify
print(df["extraction_lat_gpu_bs1"])

# Save the DataFrame to a CSV file
df.to_csv("results.csv")

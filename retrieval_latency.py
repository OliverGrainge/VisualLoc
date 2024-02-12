import faiss
import numpy as np
import time
import pandas as pd

PLATFORM = "desktop" # either embedded or 
PATH_TO_RESULTS = "/home/oliver/Documents/github/VisualLoc/Plots/PlaceRec/data/results.csv"

BACKBONES = ["vgg16", "mobilenet", "squeezenet", "efficientnet", "resnet18", "resnet50", "resnet101", "dinov2"]
AGGREGATIONS = ["mixvpr", "netvlad", "spoc", "mac", "gem"]
DESCRIPTOR_SIZES = [512, 1024, 2048, 4096]
PRECISIONS = ["int8", "fp16", "fp32"]

def compute_latency(descriptor_size, sample_size=100):
    query = np.random.randn(1, descriptor_size).astype(np.float32)
    db = np.random.randn(1000, descriptor_size).astype(np.float32)
    index = faiss.IndexFlatL2(descriptor_size)
    faiss.normalize_L2(db)
    index.add(db)
    times = []
    for _ in range(sample_size):
        st = time.time()
        _, _ = index.search(query, k=1)
        ed = time.time()
        times.append(ed - st)
    return np.mean(times) * 1000


lat_512 = compute_latency(512)
lat_1024 = compute_latency(1024)
lat_2048 = compute_latency(2048)
lat_4096 = compute_latency(4096)


results = pd.read_csv(PATH_TO_RESULTS)
results.set_index("id", inplace=True)

df_dict = {}
column_name = PLATFORM + "_retrieval_latency"
keys = []
values = []
for back in BACKBONES:
    for agg in AGGREGATIONS: 
        for d_size in DESCRIPTOR_SIZES:
            for prec in PRECISIONS:
                key = back + "_" + agg + "_" + str(d_size) + "_" + prec
                if d_size == 512:
                    value = lat_512
                elif d_size == 1024:
                    value = lat_1024
                elif d_size == 2048:
                    value = lat_2048
                elif d_size == 4096:
                    value = lat_4096

                keys.append(key)
                values.append(value)

new_data = pd.DataFrame({
    'id' : keys,
    column_name : values
})

new_data.set_index('id', inplace=True)



def update_or_add_entries(df, new_data, column_name):
    # Check if the column exists, and if not, initialize it with NaNs
    if column_name not in df.columns:
        df[column_name] = pd.NA
        
    # Convert new_data to a DataFrame for easier processing
    new_data_df = pd.DataFrame(new_data)
    
    # Merge existing df with new data based on 'id', updating existing and adding new entries
    updated_df = pd.merge(df, new_data_df, on='id', how='outer')
    
    # If the column already exists, this means there could be duplicate columns after merge (e.g., 'new_info_x' and 'new_info_y')
    # We'll coalesce these into a single column and then drop the extra
    if f'{column_name}_x' in updated_df.columns and f'{column_name}_y' in updated_df.columns:
        updated_df[column_name] = updated_df[f'{column_name}_y'].combine_first(updated_df[f'{column_name}_x'])
        updated_df.drop(columns=[f'{column_name}_x', f'{column_name}_y'], inplace=True)
    elif f'{column_name}_y' in updated_df.columns:
        # In case the merge only created 'new_info_y', rename it to the correct column name
        updated_df.rename(columns={f'{column_name}_y': column_name}, inplace=True)
    
    return updated_df


results = update_or_add_entries(results, new_data, column_name)
results.to_csv(PATH_TO_RESULTS)
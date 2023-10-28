
import numpy as np

# Example values
query_images = np.array([str(i) for i in range(100000)])
map_images = np.array([str(i) for i in range(100000)])
import numpy as np

# Create a dictionary mapping image names to a list of their indices in map_images
map_dict = {}
for idx, img in enumerate(map_images):
    map_dict.setdefault(img, []).append(idx)

# Get the indices using the dictionary
ground_truth = [map_dict.get(query, []) for query in query_images]

print(ground_truth)

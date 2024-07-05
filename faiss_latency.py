import faiss
import numpy as np
import time


def create_index_and_query(n, d):
    # Generate random data
    np.random.seed(42)  # For reproducibility
    db_vectors = np.random.rand(n, d).astype("float32")  # Database of vectors
    query_vectors = np.random.rand(1, d).astype("float32")  # Single query vector

    # Normalize vectors if using dot-product for cosine similarity (optional)
    # faiss.normalize_L2(db_vectors)
    # faiss.normalize_L2(query_vectors)

    # Create a FAISS index for the inner product (use IndexFlatIP for dot-product)
    index = faiss.IndexFlatIP(d)

    # Add vectors to the index
    index.add(db_vectors)

    # Measure the query time
    start_time = time.time()
    distances, indices = index.search(
        query_vectors, k=5
    )  # Search for the top 5 nearest neighbors
    end_time = time.time()

    # Calculate elapsed time in milliseconds
    elapsed_time_ms = (end_time - start_time) * 1000
    return elapsed_time_ms, distances, indices


# Parameters
num_embeddings = 10000  # Number of embeddings
dimension = 2048  # Dimension of each embedding

# Measurement
latency_ms, distances, indices = create_index_and_query(num_embeddings, dimension)
print(f"Matching latency: {latency_ms:.2f} ms")
print("Distances:", distances)
print("Indices:", indices)

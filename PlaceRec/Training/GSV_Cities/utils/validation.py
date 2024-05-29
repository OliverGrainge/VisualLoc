import faiss
import faiss.contrib.torch_utils
import numpy as np
from prettytable import PrettyTable
import torch
import time


def get_validation_recalls(
    r_list,
    q_list,
    k_values,
    gt,
    print_results=True,
    dataset_name="dataset without name ?",
    distance="L2",
    sparsity=None,
    descriptor_dim=None,
):
    embed_size = r_list.shape[1]
    if distance == "L2":
        faiss_index = faiss.IndexFlatL2(embed_size)
    else:
        faiss_index = faiss.IndexFlatIP(embed_size)

    # add references
    r_list = r_list / np.linalg.norm(r_list, axis=1, keepdims=True)
    q_list = q_list / np.linalg.norm(q_list, axis=1, keepdims=True)
    faiss_index.add(r_list)

    # search for queries in the index
    _, predictions = faiss_index.search(q_list, max(k_values))

    # start calculating recall_at_k
    correct_at_k = np.zeros(len(k_values))
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[q_idx])):
                correct_at_k[i:] += 1
                break

    correct_at_k = correct_at_k / len(predictions)
    d = {k: v for (k, v) in zip(k_values, correct_at_k)}

    if print_results:
        print("\n")  # print a new line
        table = PrettyTable()
        field_names = ["K"] + [str(k) for k in k_values]
        rows = ["Recall@K"] + [f"{100*v:.2f}" for v in correct_at_k]
        if sparsity is not None:
            field_names += ["Sparsity"]
            rows += [f"{sparsity:.4f}"]
        if descriptor_dim is not None:
            field_names += ["descriptor dim"]
            rows += [f"{descriptor_dim}"]
        table.field_names = field_names
        table.add_row(rows)
        print(table.get_string(title=f"Performance on {dataset_name}"))

    return d, predictions


def get_validation_recall_at_100precision(
    r_list,
    q_list,
    gt,
    print_results=True,
    dataset_name="dataset without name ?",
    distance="L2",
):
    embed_size = r_list.shape[1]
    if distance == "L2":
        faiss_index = faiss.IndexFlatL2(embed_size)
    else:
        faiss_index = faiss.IndexFlatIP(embed_size)

    # Normalize the reference and query lists
    r_list = r_list / np.linalg.norm(r_list, axis=1, keepdims=True)
    q_list = q_list / np.linalg.norm(q_list, axis=1, keepdims=True)
    faiss_index.add(r_list)

    # Search for queries in the index, querying enough results to cover all ground truths
    _, predictions = faiss_index.search(q_list, max(len(gt_i) for gt_i in gt))

    # Start calculating Recall at 100% Precision
    recalls = []
    for q_idx, pred in enumerate(predictions):
        # Find the minimum k where all top k are relevant
        k = 0
        while k < len(pred) and np.isin(pred[k], gt[q_idx]).all():
            k += 1
        # Calculate recall as the ratio of relevant items found in the top k
        recall = len(np.intersect1d(pred[:k], gt[q_idx])) / len(gt[q_idx])
        recalls.append(recall)

    average_recall = np.mean(recalls)

    if print_results:
        print("\n")  # print a new line
        print(f"Performance on {dataset_name}:")
        print(f"Average Recall@100% Precision: {100 * average_recall:.2f}%")

    return average_recall


def measure_cpu_latency(model, input_tensor, num_runs=2, warm_up=2, batch_size=1):
    """
    Measure the latency of a TorchScript PyTorch model on CPU.

    Args:
    model (torch.nn.Module): The PyTorch model to evaluate.
    input_tensor (torch.Tensor): A dummy input tensor appropriate for the model
                                 (must be the correct shape and type).
    num_runs (int): Number of times to run the model to measure latency.
    warm_up (int): Number of initial runs to warm up the model (not measured).

    Returns:
    float: The average latency in milliseconds for a single forward pass.
    """
    # Convert the model to TorchScript
    model = model.to("cpu")
    input_tensor = input_tensor.to("cpu").repeat(batch_size, 1, 1, 1)
    model_scripted = torch.jit.trace(model, input_tensor)

    # Ensure model is in evaluation mode
    model_scripted.eval()

    # Warm up runs
    with torch.no_grad():
        for _ in range(warm_up):
            _ = model_scripted(input_tensor)

    # Timing starts
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model_scripted(input_tensor)
            end_time = time.time()
            times.append(end_time - start_time)

    # Calculate average time in milliseconds
    avg_time = 1000 * sum(times) / len(times)
    print("CPU latency (TorchScript): ", avg_time)
    return avg_time


def measure_gpu_latency(model, input_tensor, num_runs=50, warm_up=20, batch_size=1):
    """
    Measure the latency of a PyTorch TorchScript model on GPU.

    Args:
    model (torch.nn.Module): The PyTorch model to evaluate.
    input_tensor (torch.Tensor): A dummy input tensor appropriate for the model
                                 (must be the correct shape and type).
    num_runs (int): Number of times to run the model to measure latency.
    warm_up (int): Number of initial runs to warm up the model (not measured).

    Returns:
    float: The average latency in milliseconds for a single forward pass.
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. GPU latency measurement cannot be performed."
        )

    model = model.to("cuda")
    input_tensor = input_tensor.to("cuda").repeat(batch_size, 1, 1, 1)

    # Convert the model to TorchScript
    model_scripted = torch.jit.trace(model, input_tensor)

    # Ensure model is in evaluation mode
    model_scripted.eval()

    # Warm up runs
    with torch.no_grad():
        for _ in range(warm_up):
            _ = model_scripted(input_tensor)
            torch.cuda.synchronize()  # Synchronize to ensure complete execution

    # Timing starts
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()  # Synchronize before starting the timer
            start_time = time.time()
            _ = model_scripted(input_tensor)
            torch.cuda.synchronize()  # Synchronize after computation to ensure all kernels have finished
            end_time = time.time()
            times.append(end_time - start_time)

    # Calculate average time in milliseconds
    avg_time = 1000 * sum(times) / len(times)
    print("GPU latency (TorchScript): ", avg_time)
    return avg_time

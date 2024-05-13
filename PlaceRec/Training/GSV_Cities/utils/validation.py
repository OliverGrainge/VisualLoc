import faiss
import faiss.contrib.torch_utils
import numpy as np
from prettytable import PrettyTable


def get_validation_recalls(
    r_list,
    q_list,
    k_values,
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
        table.field_names = ["K"] + [str(k) for k in k_values]
        table.add_row(["Recall@K"] + [f"{100*v:.2f}" for v in correct_at_k])
        print(table.get_string(title=f"Performance on {dataset_name}"))

    return d, predictions

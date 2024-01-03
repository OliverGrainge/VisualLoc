
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
import numpy as np
import faiss
import faiss.contrib.torch_utils
from prettytable import PrettyTable



def get_loss(loss_name):
    if loss_name == 'SupConLoss': return losses.SupConLoss(temperature=0.07)
    if loss_name == 'CircleLoss': return losses.CircleLoss(m=0.4, gamma=80) #these are params for image retrieval
    if loss_name == 'MultiSimilarityLoss': return losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
    if loss_name == 'ContrastiveLoss': return losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    if loss_name == 'Lifted': return losses.GeneralizedLiftedStructureLoss(neg_margin=0, pos_margin=1, distance=DotProductSimilarity())
    if loss_name == 'FastAPLoss': return losses.FastAPLoss(num_bins=30)
    if loss_name == 'NTXentLoss': return losses.NTXentLoss(temperature=0.07) #The MoCo paper uses 0.07, while SimCLR uses 0.5.
    if loss_name == 'TripletMarginLoss': return losses.TripletMarginLoss(margin=0.1, swap=False, smooth_loss=False, triplets_per_anchor='all') #or an int, for example 100
    if loss_name == 'CentroidTripletLoss': return losses.CentroidTripletLoss(margin=0.05,
                                                                            swap=False,
                                                                            smooth_loss=False,
                                                                            triplets_per_anchor="all",)
    raise NotImplementedError(f'Sorry, <{loss_name}> loss function is not implemented!')

def get_miner(miner_name, margin=0.1):
    if miner_name == 'TripletMarginMiner' : return miners.TripletMarginMiner(margin=margin, type_of_triplets="semihard") # all, hard, semihard, easy
    if miner_name == 'MultiSimilarityMiner' : return miners.MultiSimilarityMiner(epsilon=margin, distance=CosineSimilarity())
    if miner_name == 'PairMarginMiner' : return miners.PairMarginMiner(pos_margin=0.7, neg_margin=0.3, distance=DotProductSimilarity())
    return None

def get_validation_recalls(r_list, q_list, k_values, gt, print_results=True, faiss_gpu=False, dataset_name='dataset without name ?'):
        
        embed_size = r_list.shape[1]
        if faiss_gpu:
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)
        # build index
        else:
            faiss_index = faiss.IndexFlatL2(embed_size)
        
        faiss_index.add(r_list)
        _, predictions = faiss_index.search(q_list, max(k_values))
        correct_at_k = np.zeros(len(k_values))
        for q_idx, pred in enumerate(predictions):
            for i, n in enumerate(k_values):
                # if in top N then also in top NN, where NN > N
                if np.any(np.in1d(pred[:n], gt[q_idx])):
                    correct_at_k[i:] += 1
                    break
        correct_at_k = correct_at_k / len(predictions)
        d = {k:v for (k,v) in zip(k_values, correct_at_k)}

        if print_results:
            print('\n') # print a new line
            table = PrettyTable()
            table.field_names = ['K']+[str(k) for k in k_values]
            table.add_row(['Recall@K']+ [f'{100*v:.2f}' for v in correct_at_k])
            print(table.get_string(title=f"Performance on {dataset_name}"))
        
        return d, predictions
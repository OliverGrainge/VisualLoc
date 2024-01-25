
import torch 
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from parsers import train_arguments

args = train_arguments()

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res




class RkdDistance(nn.Module):        
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss


class TeacherAverageLoss(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.distance_loss = RkdDistance()

    def forward(self, student_desc: torch.tensor, teacher_desc: torch.tensor):
        loss = torch.stack([self.distance_loss(student_desc, desc) for desc in teacher_desc]).mean()
        return loss


args = train_arguments()
loss = TeacherAverageLoss(args)
"""
config = get_config()

DATASET_NAME = "san_francisco"

datasets_dir = join(config["datasets_directory"], "datasets_vg", "datasets", DATASET_NAME, "images", "test")
database_folder = join(datasets_dir, "database")
print(database_folder)
queries_folder = join(datasets_dir, "queries")
database_paths = sorted(glob(database_folder + "/*.jpg"))
queries_paths = sorted(glob(queries_folder + "/*.jpg"))

database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in database_paths]).astype(float)
queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in queries_paths]).astype(float)
print(len(queries_utms))

knn = NearestNeighbors()
knn.fit(database_utms)
gt = knn.radius_neighbors(queries_utms,
                                     radius=25,
                                     return_distance=False)

idx_zero = []
for i, pos in enumerate(gt):
    if len(pos) == 0:
        idx_zero.append(i)

if not len(idx_zero) == 0:
    queries_paths = np.delete(queries_paths, np.array(idx_zero))
    gt = np.delete(gt, np.array(idx_zero))

assert len(queries_paths) == gt.shape[0]

queries_paths = np.array([q_pth.replace(join(config["datasets_directory"], "datasets_vg", "datasets", DATASET_NAME), "") for q_pth in queries_paths])
database_paths = np.array([d_pth.replace(join(config["datasets_directory"], "datasets_vg", "datasets", DATASET_NAME), "") for d_pth in database_paths])

print(len(queries_paths), len(database_paths))
def save_dataset(name, query_paths, database_paths, ground_truth):
    np.save("tmp/" + DATASET_NAME + "/" + DATASET_NAME + "_qImages.npy", query_paths)
    np.save("tmp/" + DATASET_NAME + "/" + DATASET_NAME + "_dbImages.npy", database_paths)
    np.save("tmp/" + DATASET_NAME + "/" + DATASET_NAME + "_gt.npy", ground_truth)


save_dataset(DATASET_NAME, queries_paths, database_paths, gt)

"""
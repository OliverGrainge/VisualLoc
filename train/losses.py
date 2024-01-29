
import torch 
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from typing import List

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


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x):
        return x / x.norm(p=2, dim=1, keepdim=True)


def simple_mlp(in_dim: int, out_dim: int):
    net = nn.Sequential(nn.Linear(in_dim, 2048),
                        nn.ReLU(),
                        nn.Linear(2048, 2048), 
                        nn.ReLU(),
                        nn.Linear(2048, out_dim), 
                        Normalize())
    return net


################################ Multi-Teacher Distillation Losses ################################

class TeacherAverageLoss(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.distance_loss = RkdDistance()

    def forward(self, student_desc: torch.tensor, teacher_desc: List[torch.tensor]):
        loss = torch.stack([self.distance_loss(student_desc, desc) for desc in teacher_desc]).mean()
        return loss

class TeacherWeightedAverageLoss(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(len(args.teacher_methods))/len(args.teacher_methods))
        self.distance_loss = RkdDistance()

    def forward(self, student_desc: torch.tensor, teacher_desc: torch.tensor):
        losses = torch.stack([self.distance_loss(student_desc, desc) for desc in teacher_desc]).flatten()
        losses = losses * F.softmax(self.weights)
        return losses.sum()


class TeacherFeedLoss(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.distance_loss = RkdDistance()
        self.feed_heads = [simple_mlp(args.student_features_dim, 2048) for _ in range(len(args.teacher_methods))]
        if args.device == "cuda":
            self.feed_heads = [head.to("cuda") for head in self.feed_heads]

    def forward(self, student_desc: torch.tensor, teacher_desc: torch.tensor):
        all_student_desc = [mlp(student_desc) for mlp in self.feed_heads]
        loss = torch.stack([self.distance_loss(s_desc, t_desc) for s_desc, t_desc in zip(all_student_desc, teacher_desc)])
        return loss.mean()


class TeacherAdaptiveLoss(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.distance_loss = RkdDistance()
        self.adaption_heads = [nn.Linear(t_dim, args.student_features_dim).to(args.device) for t_dim in args.teacher_features_dim]
        self.weight_head = nn.Linear(args.student_features_dim, 1).to(args.device)

    def forward(self, student_desc: torch.tensor, teacher_desc: torch.tensor):
        teacher_features = [mlp(t_desc) for mlp, t_desc in zip(self.adaption_heads, teacher_desc)] # maps the teacher features to same shape as student features
        fused_features = [student_desc * x for x in teacher_features] # performs hadamard product with teacher features 
        weights = torch.stack([self.weight_head(x) for x in fused_features]).flatten()
        weights = F.softmax(weights)
        loss = torch.stack([weights[i] * self.distance_loss(student_desc, t_desc) for i, t_desc in enumerate(teacher_desc)]).sum()
        return loss 


def get_multi_teacher_loss(args: Namespace):
    if args.fusion_method == "average":
        return TeacherAverageLoss(args)
    elif args.fusion_method == "weighted_average":
        return TeacherWeightedAverageLoss(args)
    elif args.fusion_method == "feed":
        return TeacherFeedLoss(args)
    elif args.fusion_method == "adaptive":
        return TeacherAdaptiveLoss(args)
    else: 
        raise NotImplementedError()








    


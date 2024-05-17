import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(KDLoss, self).__init__()
        self.temperature = temperature

    def forward(self, student_descriptors, teacher_descriptors, miner_outputs=None):
        student_descriptors = F.normalize(student_descriptors, p=2, dim=1)
        teacher_descriptors = F.normalize(teacher_descriptors, p=2, dim=1)
        student_logits = student_descriptors / self.temperature
        teacher_logits = teacher_descriptors / self.temperature
        log_student_softmax = F.log_softmax(student_logits, dim=1)
        teacher_softmax = F.softmax(teacher_logits, dim=1)
        kd_loss = F.kl_div(
            log_student_softmax, teacher_softmax, reduction="batchmean"
        ) * (self.temperature**2)
        return kd_loss


class KDMSELoss(nn.Module):
    def __init__(self):
        super(KDMSELoss, self).__init__()

    def forward(self, student_descriptors, teacher_descriptors, miner_outputs=None):
        student_descriptors = F.normalize(student_descriptors, p=2, dim=1)
        teacher_descriptors = F.normalize(teacher_descriptors, p=2, dim=1)
        mse_loss = F.mse_loss(
            student_descriptors, teacher_descriptors, reduction="mean"
        )
        return mse_loss


class RKdAngle(nn.Module):
    def forward(self, student, teacher, miner_outputs=None):
        with torch.no_grad():
            td = teacher.unsqueeze(0) - teacher.unsqueeze(1)
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = student.unsqueeze(0) - student.unsqueeze(1)
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
        loss = F.smooth_l1_loss(s_angle, t_angle, reduction="mean")
        return loss


class MultiSimilarityLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=50.0, base=0.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base

    def forward(self, student, miner_outputs=None):
        student = F.normalize(student, p=2, dim=1)
        S = student @ student.T
        pos_exp = self.apply_margin(S, self.base)
        neg_exp = self.apply_margin(self.base, S)
        pos_mask, neg_mask = self.get_masks(S, miner_outputs)
        pos_loss = (1.0 / self.alpha) * self.logexpsum(
            self.alpha * pos_exp, keep_mask=pos_mask.bool(), add_one=True
        )
        neg_loss = (1.0 / self.beta) * self.logsumexp(
            self.beta * neg_exp, keep_mask=neg_mask.bool(), add_one=True
        )
        return pos_loss + neg_loss

    def logsumexp(x, keep_mask=None, add_one=True, dim=1):
        if keep_mask is not None:
            x = x.masked_fill(~keep_mask, torch.finfo(x.dtype).min)
        if add_one:
            zeros = torch.zeros(
                x.size(dim - 1), dtype=x.dtype, device=x.device
            ).unsqueeze(dim)
            x = torch.cat([x, zeros], dim=dim)

        output = torch.logsumexp(x, dim=dim, keepdim=True)
        if keep_mask is not None:
            output = output.masked_fill(~torch.any(keep_mask, dim=dim, keepdim=True), 0)
        return output

    def get_masks(self, similarity, miner_outputs):
        a1, p, a2, n = miner_outputs
        pos_mask, neg_mask = torch.zeros_like(similarity), torch.zeros_like(similarity)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1
        return pos_mask, neg_mask

    def apply_margin(base, margin):
        return base - margin


class MultiSimilarityWeightedRKdAngle(nn.Module):
    def __init__(self, alpha=1.0, beta=50.0, base=0.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base

    def forward(self, student, teacher, miner_outputs=None):
        student = F.normalize(student, p=2, dim=1)
        teacher = F.normalize(teacher, p=2, dim=1)
        S = student @ student.T
        pos_mask, neg_mask = self.get_masks(S, miner_outputs[pos_mask.bool()])
        print(miner_outputs[0][:20])
        raise Exception

        """
        pos_exp = self.apply_margin(S, self.base)
        neg_exp = self.apply_margin(self.base, S)
        pos_mask, neg_mask = self.get_masks(S, miner_outputs)
        pos_loss = (1.0/self.alpha) * self.logexpsum(
            self.alpha * pos_exp, keep_mask=pos_mask.bool(), add_one=True
        )
        neg_loss = (1.0/self.beta) * self.logsumexp(
            self.beta * neg_exp, keep_mask=neg_mask.bool(), add_one=True
        )
        return pos_loss + neg_loss
        """

    def logsumexp(x, keep_mask=None, add_one=True, dim=1):
        if keep_mask is not None:
            x = x.masked_fill(~keep_mask, torch.finfo(x.dtype).min)
        if add_one:
            zeros = torch.zeros(
                x.size(dim - 1), dtype=x.dtype, device=x.device
            ).unsqueeze(dim)
            x = torch.cat([x, zeros], dim=dim)

        output = torch.logsumexp(x, dim=dim, keepdim=True)
        if keep_mask is not None:
            output = output.masked_fill(~torch.any(keep_mask, dim=dim, keepdim=True), 0)
        return output

    def get_masks(self, similarity, miner_outputs):
        a1, p, a2, n = miner_outputs
        pos_mask, neg_mask = torch.zeros_like(similarity), torch.zeros_like(similarity)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1
        return pos_mask, neg_mask

    def apply_margin(base, margin):
        return base - margin


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
    def forward(self, student, teacher, miner_outputs=None):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction="mean")
        return loss


def get_kd_loss(kd_loss_name):
    if kd_loss_name.lower() == "rkddistance":
        return RkdDistance()
    elif kd_loss_name.lower() == "rkdangle":
        return RKdAngle()
    elif kd_loss_name.lower() == "mse":
        return KDMSELoss()
    elif kd_loss_name.lower() == "kld":
        return KDLoss()
    elif kd_loss_name.lower() == "multisimilarityweightsrkdangle":
        return MultiSimilarityWeightedRKdAngle()
    else:
        raise NotImplementedError

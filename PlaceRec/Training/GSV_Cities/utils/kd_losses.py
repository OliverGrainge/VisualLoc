import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(KDLoss, self).__init__()
        self.temperature = temperature

    def forward(self, student_descriptors, teacher_descriptors):
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

    def forward(self, student_descriptors, teacher_descriptors):
        student_descriptors = F.normalize(student_descriptors, p=2, dim=1)
        teacher_descriptors = F.normalize(teacher_descriptors, p=2, dim=1)
        mse_loss = F.mse_loss(
            student_descriptors, teacher_descriptors, reduction="mean"
        )
        return mse_loss


class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            td = teacher.unsqueeze(0) - teacher.unsqueeze(1)
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = student.unsqueeze(0) - student.unsqueeze(1)
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
        loss = F.smooth_l1_loss(s_angle, t_angle, reduction="mean")
        return loss


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
    elif kd_loss_name.lower() == "rdkangle":
        return RKdAngle()
    elif kd_loss_name.lower() == "mse":
        return KDMSELoss()
    elif kd_loss_name.lower() == "kld":
        return KDLoss()
    else:
        raise NotImplementedError

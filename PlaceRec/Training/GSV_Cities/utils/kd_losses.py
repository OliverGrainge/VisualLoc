import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidedMultiSimilarity(nn.Module):
    def __init__(self, alpha=1.0, beta=50.0, base=0.0):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.base = base

    def forward(self, student, teacher, miner_outputs):
        student_desc = F.normalize(student, p=2, dim=1)
        teacher_desc = F.normalize(teacher, p=2, dim=1)

        ST = teacher_desc @ teacher_desc.T
        # ST = (ST + 1)/2
        S = student_desc @ student_desc.T
        pos_exp = self.apply_margin(S, self.base)
        neg_exp = self.apply_margin(self.base, S)

        pos_mask, neg_mask = self.get_masks(S, miner_outputs)

        pos_loss = (1.0 / self.alpha) * self.logsumexp(
            self.alpha * pos_exp, keep_mask=pos_mask, add_one=True, teacher_sim=ST
        )
        neg_loss = (1.0 / self.beta) * self.logsumexp(
            self.beta * neg_exp,
            keep_mask=neg_mask.bool(),
            add_one=True,
            teacher_sim=-ST,
        )
        return torch.mean(pos_loss + neg_loss)

    def get_masks(self, similarity, miner_outputs):
        a1, p, a2, n = miner_outputs
        pos_mask, neg_mask = torch.zeros_like(similarity), torch.zeros_like(similarity)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1
        return pos_mask.bool(), neg_mask.bool()

    def apply_margin(self, x, margin):
        return margin - x

    def logsumexp(self, x, keep_mask=None, add_one=True, dim=1, teacher_sim=None):
        if keep_mask is not None:
            x = x.masked_fill(~keep_mask, torch.finfo(x.dtype).min)
            x[keep_mask] *= torch.exp(teacher_sim)[keep_mask]
        if add_one:
            zeros = torch.zeros(
                x.size(dim - 1), dtype=x.dtype, device=x.device
            ).unsqueeze(dim)
            x = torch.cat([x, zeros], dim=dim)

        output = torch.logsumexp(x, dim=dim, keepdim=True)
        if keep_mask is not None:
            output = output.masked_fill(~torch.any(keep_mask, dim=dim, keepdim=True), 0)
        return output


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


"""
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
"""


class RKdAngle(nn.Module):
    def forward(self, student, teacher, miner_outputs=None):
        with torch.no_grad():
            teacher = F.normalize(teacher, p=2, dim=1)
            S_teacher = teacher @ teacher.T
        student = F.normalize(student, p=2, dim=1)
        S_student = student @ student.T
        loss = F.smooth_l1_loss(
            S_student.flatten(), S_teacher.flatten(), reduction="mean"
        )
        return loss


class RegressionRKdAngle(nn.Module):
    def __init__(self):
        super().__init__()
        self.angle_loss = RKdAngle()
        self.regresssion_loss = KDMSELoss()

    def forward(self, student, teacher, miner_outputs=None):
        l1 = self.angle_loss(student, teacher, miner_outputs)
        l2 = self.regresssion_loss(student, teacher, miner_outputs)

        return l1 + (4 * l2)


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


class MultiSimilarityWeighted(nn.Module):
    def __init__(self, alpha=1.0, beta=50.0, base=0.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base

    def compute_weights(self, S, miner_outputs):
        (
            a1,
            p,
            a2,
            n,
        ) = miner_outputs  # Assuming these are arrays of indices for anchors, positives, negatives.

        # Get masks
        pos_mask, neg_mask = self.get_masks(S, miner_outputs)

        # Positive weights
        term1 = torch.exp(-self.alpha * (self.base - S[a1, p]))
        term2 = (
            torch.exp(-self.alpha * (S[a1] - S[a1, p].unsqueeze(1)))
            * pos_mask[a1].float()
        )
        pos_weights = 1 / (term1 + term2.sum(dim=1))

        # Negative weights
        numerator = torch.exp(self.beta * (S[a2, n] - self.base))
        denominator = (
            torch.exp(self.beta * (S[a2, :] - self.base)) * neg_mask[a2].float()
        )
        neg_weights = numerator / (1 + denominator.sum(dim=1))

        # Concatenate weights
        weights = torch.cat((pos_weights, neg_weights))

        return weights

    """
    def compute_weights(self, S, miner_outputs=None):

        pos_mask, neg_mask = self.get_masks(S, miner_outputs)

        # positive weights
        term1 = torch.exp(-self.alpha * self.apply_margin(self.base, S[pos_mask]))
        term2 = torch.exp(-self.alpha * self.apply_margin(S[miner_outputs[0]], S[pos_mask].view(-1, 1)))
        term2 = term2 * pos_mask[miner_outputs[0]].float() 
        mask = pos_mask[miner_outputs[0]].float()
        term2 = term2 * mask
        term2 = torch.sum(term2, dim=1)
        pos_weights = 1/(term1 + term2)

        # negative weights 
        numerator = torch.exp(self.beta*self.apply_margin(S[neg_mask], self.base))
        denominator = torch.exp(self.beta * self.apply_margin(S[miner_outputs[2]], self.base))
        mask = neg_mask[miner_outputs[2]].float()
        denominator = denominator * mask
        denominator = torch.sum(denominator, dim=1)
        neg_weights = numerator/(1+denominator)

        # compute loss
        return torch.concat((pos_weights, neg_weights))
    """

    def get_masks(self, similarity, miner_outputs):
        a1, p, a2, n = miner_outputs
        pos_mask, neg_mask = torch.zeros_like(similarity), torch.zeros_like(similarity)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1
        return pos_mask.bool(), neg_mask.bool()

    def apply_margin(self, x, margin):
        return x - margin


class MultiSimilarityWeightedRKdAngle_Huber(MultiSimilarityWeighted):
    def __init__(self, alpha=1.0, beta=50.0, base=0.0):
        super().__init__(alpha=alpha, beta=beta, base=base)

    def forward(self, student, teacher, miner_outputs=None):
        if miner_outputs == None:
            raise Exception("MutiSimilarityWeighted functions must have miner pairs")
        student = F.normalize(student, p=2, dim=1)
        teacher = F.normalize(teacher, p=2, dim=1)
        teacher_S = teacher @ teacher.T
        S = student @ student.T
        weights = self.compute_weights(S, miner_outputs)
        pos_mask, neg_mask = self.get_masks(S, miner_outputs)
        teacher_relations = torch.concat((teacher_S[pos_mask], teacher_S[neg_mask]))
        student_relations = torch.concat((S[pos_mask], S[neg_mask]))
        loss = F.smooth_l1_loss(student_relations, teacher_relations, reduction="none")
        scaled_loss = loss * weights
        loss = torch.mean(scaled_loss)
        return loss


class MultiSimilarityWeightedRKdAngle_MSE(MultiSimilarityWeighted):
    def __init__(self, alpha=1.0, beta=50.0, base=0.0):
        super().__init__(alpha=alpha, beta=beta, base=base)

    def forward(self, student, teacher, miner_outputs=None):
        if miner_outputs == None:
            raise Exception("MutiSimilarityWeighted functions must have miner pairs")
        student = F.normalize(student, p=2, dim=1)
        teacher = F.normalize(teacher, p=2, dim=1)
        teacher_S = teacher @ teacher.T
        S = student @ student.T
        weights = self.compute_weights(S, miner_outputs)
        pos_mask, neg_mask = self.get_masks(S, miner_outputs)
        teacher_relations = torch.concat((teacher_S[pos_mask], teacher_S[neg_mask]))
        student_relations = torch.concat((S[pos_mask], S[neg_mask]))
        loss = F.mse_loss(student_relations, teacher_relations, reduction="none")
        scaled_loss = loss * weights
        loss = torch.mean(scaled_loss)
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


def pdist(A, B, squared=False, eps=1e-12):
    D = A.pow(2).sum(1) + (-2) * B.mm(A.t())
    D = (B.pow(2).sum(1) + D.t()).clamp(min=eps)

    if not squared:
        D = D.sqrt()

    if torch.equal(A, B):
        D = D.clone()
        D[range(len(A)), range(len(A))] = 0

    return D


class Relaxed_Contra(nn.Module):
    """Relaxed Contrative loss function."""

    def __init__(self, sigma=1, delta=1):
        super(Relaxed_Contra, self).__init__()

        self.sigma = sigma
        self.delta = delta

    def forward(self, s_emb, t_emb, miner_outputs=None):
        s_emb = F.normalize(s_emb, p=2, dim=1)

        T_dist = pdist(t_emb, t_emb, False)
        dist_mean = T_dist.mean(1, keepdim=True)
        T_dist = T_dist / dist_mean

        with torch.no_grad():
            S_dist = pdist(s_emb, s_emb, False)
            P = torch.exp(-S_dist.pow(2) / self.sigma)

        pos_weight = P
        neg_weight = 1 - P

        pull_losses = torch.relu(T_dist).pow(2) * pos_weight
        push_losses = torch.relu(self.delta - T_dist).pow(2) * neg_weight

        pull_losses = pull_losses[T_dist > 0]
        push_losses = push_losses[T_dist > 0]
        loss = (pull_losses.sum() + push_losses.sum()) / len(t_emb)

        return loss


class Relaxed_MS(nn.Module):
    """Relaxed MS loss function."""

    def __init__(self, sigma=1, delta=1):
        super(Relaxed_MS, self).__init__()

        self.sigma = sigma
        self.delta = delta
        self.scale_pos = 1
        self.scale_neg = 2

    def forward(self, s_emb, t_emb, miner_outputs=None):
        batch_size = t_emb.size(0)  # Assuming t_emb is your tensor of embeddings

        # Create student_col as a range of indices
        student_col = torch.arange(0, batch_size).tolist()

        s_emb = F.normalize(s_emb, p=2, dim=1)

        losses = []
        T_dist = pdist(t_emb, t_emb, False)
        T_dist = T_dist / T_dist.mean(1, keepdim=True)

        S_dist = pdist(s_emb, s_emb, False)
        P = torch.exp(-S_dist.pow(2) / self.sigma)

        batch_size = len(student_col) // 2
        for i in range(batch_size):
            P[i, i + batch_size] = 1
            P[i + batch_size, i] = 1

        for i in range(len(student_col)):
            dist_i = torch.cat([T_dist[i][0:i], T_dist[i][i + 1 :]])
            P_i = torch.cat([P[i][0:i], P[i][i + 1 :]])

            pos_weight = P_i
            neg_weight = 1 - P_i

            pos_exp = pos_weight * torch.exp(self.scale_pos * (dist_i))
            neg_exp = neg_weight * torch.exp(self.scale_neg * (self.delta - dist_i))

            P_sim_sum = pos_exp.sum()
            N_sim_sum = neg_exp.sum()

            pulling_loss = 1 / self.scale_pos * torch.log(1 + P_sim_sum)
            pushing_loss = 1 / self.scale_neg * torch.log(1 + N_sim_sum)

            losses.append(pulling_loss + pushing_loss)

        losses = torch.stack(losses)
        loss = losses[losses > 0].mean()

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
    elif kd_loss_name.lower() == "multisimilarityweightedrkdanglehuber":
        return MultiSimilarityWeightedRKdAngle_Huber()
    elif kd_loss_name.lower() == "multisimilarityweightedrkdanglemse":
        return MultiSimilarityWeightedRKdAngle_MSE()
    elif kd_loss_name.lower() == "regressionrkdangle":
        return RegressionRKdAngle()
    elif kd_loss_name.lower() == "relaxedms":
        return Relaxed_MS()
    elif kd_loss_name.lower() == "relaxedcont":
        return Relaxed_Contra()
    elif kd_loss_name.lower() == "guidedmultisimilarity":
        return GuidedMultiSimilarity()
    else:
        raise NotImplementedError

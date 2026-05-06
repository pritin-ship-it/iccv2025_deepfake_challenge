import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEWithLogitsLoss_LS(nn.Module):
    def __init__(self, label_smoothing=0.1, reduction='mean'):
        super(BCEWithLogitsLoss_LS, self).__init__()
        assert 0 <= label_smoothing < 1, "label_smoothing value must be between 0 and 1."
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, target):
        if self.label_smoothing > 0:
            positive_smoothed_labels = 1.0 - self.label_smoothing
            negative_smoothed_labels = self.label_smoothing
            target = target * positive_smoothed_labels + \
                (1 - target) * negative_smoothed_labels

        loss = self.bce_with_logits(input, target)
        return loss

def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2, loss_multip=1):
    ce_loss = torch.nn.CrossEntropyLoss(reduction="none")(inputs, targets)

    p = torch.softmax(inputs, dim=1)
    p = torch.gather(p, dim=1, index=targets.unsqueeze(dim=1))[:, 0].detach()

    loss = ce_loss * ((1 - p) ** gamma)

    if alpha >= 0 and 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    loss = loss.mean()
    return loss * loss_multip


def dice_loss(input, target):
    smooth = 1.

    iflat = input.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (torch.abs(iflat * tflat)).sum()
    
    return 1 - ((2. * intersection + smooth) /
              ((iflat*iflat).sum() + (tflat*tflat).sum() + smooth))


def get_loss(labels_pred, labels_true, criterion, aux_pred=None, aux_true=None, criterion_mh=None):
    res = {}
    res["BCE_loss"] = criterion(labels_pred, labels_true)
    if criterion_mh:
        res["CE_loss"] = criterion_mh(aux_pred, aux_true) * 0.001

    return res
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_with_logits(inputs, targets.float())
        p_t = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t)**self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__(); self.T = temperature
    def forward(self, feats, labels):
        feats = F.normalize(feats, dim=1)
        sim   = torch.mm(feats, feats.t()) / self.T          # NxN
        # mask positives (same label) & self‐contrast
        mask  = labels.unsqueeze(0) == labels.unsqueeze(1)
        sim_exp = torch.exp(sim) * (~torch.eye(len(feats),dtype=bool,device=feats.device))
        pos_exp = sim_exp * mask
        loss = -torch.log( pos_exp.sum(1) / sim_exp.sum(1) ).mean()
        return loss
    
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, logits, targets):
        pt = torch.sigmoid(logits).detach()
        focal = (1-pt).pow(self.gamma)
        loss  = self.bce(logits, targets.float()) * focal
        alpha = self.alpha if targets.sum() else (1-self.alpha)
        return (alpha * loss).mean()
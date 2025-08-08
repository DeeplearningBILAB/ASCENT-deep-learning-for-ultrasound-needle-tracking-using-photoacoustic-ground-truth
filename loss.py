import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        targets = targets.float()
        logpt = -self.bce_with_logits(logits, targets)
        pt = torch.exp(logpt)

        focal_term = self.alpha * (1 - pt)**self.gamma if self.alpha is not None else (1 - pt)**self.gamma
        loss = -focal_term * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else: # 'none'
            return loss

class UnifiedLoss(nn.Module):
    def __init__(self, w_bce=1.0, w_dice=1.0, w_focal=0.5, aux_weight=0.4, focal_alpha=0.8, focal_gamma=2.0):
        super(UnifiedLoss, self).__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.aux_weight = aux_weight

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def _calculate_loss(self, logits, targets, use_focal=False):
        loss_bce = self.bce_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)
        loss = self.w_bce * loss_bce + self.w_dice * loss_dice
        if use_focal:
            loss_focal = self.focal_loss(logits, targets)
            loss += self.w_focal * loss_focal
        return loss

    def forward(self, outputs, targets):
        d0, d1, d2, d3, d4, d5, d6 = outputs

        targets = targets.float()

        # Main loss
        loss_main = self._calculate_loss(d0, targets, use_focal=True)

        # Auxiliary losses (BCE + Dice only)
        loss_aux = 0
        for aux_output in [d1, d2, d3, d4, d5, d6]:
            loss_aux += self._calculate_loss(aux_output, targets, use_focal=False)

        total_loss = loss_main + self.aux_weight * loss_aux

        return total_loss, loss_main

def muti_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    unified_loss = UnifiedLoss()
    total_loss, loss_main = unified_loss((d0, d1, d2, d3, d4, d5, d6), labels_v)
    return loss_main, total_loss

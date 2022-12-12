import torch.nn as nn
import torch.nn.functional as F
import torch 

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class EvidentialLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(EvidentialLoss, self).__init__()
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        # Inputs are Dirichlet parameters alpha_0 ... alpha_C
        # Targets are class indices

        num_classes = inputs.shape[1]
        loss = 0

        for input, target in zip(inputs, targets):
            ignore_px = target == self.ignore_index
            if ignore_px.all():
                continue

            target[ignore_px] = 0

            target_probs = F.one_hot(target, num_classes=num_classes).permute(2, 0, 1)

            strength = input.sum(axis=0)
            probs = input / (strength + 1e-12)

            L_err = ((probs - target_probs) ** 2).sum(axis=0)
            L_var = (probs * (1 - probs) / (strength + 1)).sum(axis=0)

            loss += (L_err + L_var)[~ignore_px].mean()

            if torch.isnan(loss).any():
                breakpoint()
        
        return loss

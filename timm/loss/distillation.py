import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temp: float):
        super(DistillationLoss, self).__init__()
        self.T = temp

    def forward(self, out1, out2):
        loss = F.kl_div(
            F.log_softmax(out1 / self.T, dim=1),
            F.softmax(out2 / self.T, dim=1),
            reduction="none",
        )

        return loss


def stain_Distillation(self, weights, kd_loss,fg_para_0):
    b = (torch.sum(torch.Tensor(weights).cuda() * torch.mean(kd_loss, dim=1))) / (
        torch.sum(torch.Tensor(weights))).cuda()
    kd_loss = b.repeat(fg_para_0.shape[0])
    return kd_loss
import torch.nn.functional as f
import pytorch_ssim
from piqa import SSIM


def total_loss(y_true, y_pred):
    l1_loss = f.l1_loss(y_pred, y_true)
    ssim_loss = -pytorch_ssim.ssim(y_pred, y_true)
    return l1_loss + ssim_loss


class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from pytorch_msssim import ssim


class AdvancedLoss(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device
        self.l1 = nn.L1Loss()
        self.lpips = lpips.LPIPS(net='vgg').to(device)

    def color_loss(self, pred, gt):
        pred_mean = torch.mean(pred, dim=[2, 3])
        gt_mean = torch.mean(gt, dim=[2, 3])
        return F.mse_loss(pred_mean, gt_mean)

    def edge_loss(self, pred, gt):

        sobel_x = torch.tensor(
            [[1,0,-1],[2,0,-2],[1,0,-1]],
            dtype=torch.float32,
            device=self.device
        ).view(1,1,3,3)

        sobel_y = torch.tensor(
            [[1,2,1],[0,0,0],[-1,-2,-1]],
            dtype=torch.float32,
            device=self.device
        ).view(1,1,3,3)

        sobel_x = sobel_x.repeat(3,1,1,1)
        sobel_y = sobel_y.repeat(3,1,1,1)

        pred_edge_x = F.conv2d(pred, sobel_x, padding=1, groups=3)
        pred_edge_y = F.conv2d(pred, sobel_y, padding=1, groups=3)
        gt_edge_x = F.conv2d(gt, sobel_x, padding=1, groups=3)
        gt_edge_y = F.conv2d(gt, sobel_y, padding=1, groups=3)

        pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-6)
        gt_edge = torch.sqrt(gt_edge_x**2 + gt_edge_y**2 + 1e-6)

        return F.l1_loss(pred_edge, gt_edge)

    def ssim_loss(self, pred, gt):
        return 1 - ssim(pred, gt, data_range=1.0)

    def forward(self, pred, gt):

        loss_l1 = self.l1(pred, gt)
        loss_lpips = self.lpips(pred * 2 - 1, gt * 2 - 1).mean()
        loss_color = self.color_loss(pred, gt)
        loss_edge = self.edge_loss(pred, gt)
        loss_ssim = self.ssim_loss(pred, gt)

        total_loss = (
            1.0 * loss_l1 +
            0.5 * loss_lpips +
            0.3 * loss_color +
            0.3 * loss_edge +
            0.5 * loss_ssim
        )

        return total_loss, {
            "L1": loss_l1.item(),
            "LPIPS": loss_lpips.item(),
            "Color": loss_color.item(),
            "Edge": loss_edge.item(),
            "SSIM": loss_ssim.item()
        }

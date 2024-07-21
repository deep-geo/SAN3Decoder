import torch
from torch import nn as nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        num_pos = torch.sum(mask)
        num_neg = mask.numel() - num_pos
        w_pos = (1 - p) ** self.gamma
        w_neg = p ** self.gamma

        loss_pos = -self.alpha * mask * w_pos * torch.log(p + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - p + 1e-7)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-7)

        return loss



class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        intersection = torch.sum(p * mask)
        union = torch.sum(p) + torch.sum(mask)
        dice_loss = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice_loss

class MaskIoULoss(nn.Module):

    def __init__(self, ):
        super(MaskIoULoss, self).__init__()

    def forward(self, pred_mask, ground_truth_mask, pred_iou):
        """
        pred_mask: [B, 1, H, W]
        ground_truth_mask: [B, 1, H, W]
        pred_iou: [B, 1]
        """
        assert pred_mask.shape == ground_truth_mask.shape, "pred_mask and ground_truth_mask should have the same shape."

        p = torch.sigmoid(pred_mask)
        intersection = torch.sum(p * ground_truth_mask)
        union = torch.sum(p) + torch.sum(ground_truth_mask) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        iou_loss = torch.mean((iou - pred_iou) ** 2)
        return iou_loss


class FocalDiceloss_IoULoss(nn.Module):
    
    def __init__(self, weight=20.0, iou_scale=1.0):
        super(FocalDiceloss_IoULoss, self).__init__()
        self.weight = weight
        self.iou_scale = iou_scale
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.maskiou_loss = MaskIoULoss()

    def forward(self, pred, mask, pred_iou):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        focal_loss = self.focal_loss(pred, mask)
        dice_loss =self.dice_loss(pred, mask)
        loss1 = self.weight * focal_loss + dice_loss
        loss2 = self.maskiou_loss(pred, mask, pred_iou)
        loss = loss1 + loss2 * self.iou_scale
        return loss

class Total_Loss(nn.Module):
    def __init__(self, weight=20.0, iou_scale=1.0):
        super(Total_Loss, self).__init__()
        self.segmentation_loss = FocalDiceloss_IoULoss(weight, iou_scale)
        self.edge_loss = FocalDiceloss_IoULoss(weight, iou_scale)
        self.cluster_edge_loss = FocalDiceloss_IoULoss(weight, iou_scale)

    def forward(self, seg_pred, seg_mask, seg_iou_pred,
                edge_pred, edge_mask, edge_iou_pred,
                cluster_edge_pred, cluster_edge_mask, cluster_edge_iou_pred,
                weights=None):
        """
        Combines the losses from segmentation, normal edge, and cluster edge decoders.
        """
        seg_loss = self.segmentation_loss(seg_pred, seg_mask, seg_iou_pred)
        edge_loss = self.edge_loss(edge_pred, edge_mask, edge_iou_pred)
        cluster_edge_loss = self.cluster_edge_loss(cluster_edge_pred, cluster_edge_mask, cluster_edge_iou_pred)

        total_loss = 1.0*seg_loss + 0.0*edge_loss + 0.0*cluster_edge_loss
        return total_loss


class BCE_Loss(nn.Module):

    def __init__(self):
        super(BCE_Loss, self).__init__()

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W] - Predictions as logits
        mask: [B, 1, H, W] - Ground truth masks
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)  # Apply sigmoid to convert logits to probabilities

        # Compute binary cross-entropy loss
        bce_loss = F.binary_cross_entropy(p, mask, reduction='mean')
        return bce_loss


class BCE_Diceloss_IoULoss(nn.Module):

    def __init__(self, weight=1.0, iou_scale=1.0):
        super(BCE_Diceloss_IoULoss, self).__init__()
        self.weight = weight
        self.iou_scale = iou_scale
        self.bce_loss = BCE_Loss()
        self.dice_loss = DiceLoss()
        self.maskiou_loss = MaskIoULoss()

    def forward(self, pred, mask, pred_iou):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        BCE_loss = self.BCE_loss(pred, mask)
        dice_loss =self.dice_loss(pred, mask)
        loss1 = BCE_loss + dice_loss
        #loss2 = self.maskiou_loss(pred, mask, pred_iou)
        #loss = loss1 #+ loss2 * self.iou_scale
        return loss1


class SensitivityLoss(nn.Module):
    def __init__(self, gamma0=0.5, gamma1=0):
        super(SensitivityLoss, self).__init__()
        self.gamma0 = gamma0
        self.gamma1 = gamma1

    def forward(self, pred, mask, weights=None):
        """
        Compute Sensitivity Loss.
        
        Args:
            pred (torch.Tensor): Predicted logits from the model of shape [B, 1, H, W].
            mask (torch.Tensor): Ground truth binary mask of shape [B, 1, H, W].
            weights (torch.Tensor, optional): Sample weights of shape [B]. Default is None.
        
        Returns:
            torch.Tensor: Computed Sensitivity Loss.
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        # Convert logits to probabilities
        p = torch.sigmoid(pred)
        
        epsilon = 1e-12  # Small constant for numerical stability

        # Calculate the weighting factors for positive and negative samples
        w_pos = (1 - p) ** (-self.gamma0)
        w_neg = p ** (-self.gamma1)

        # Calculate the sensitivity loss for positive and negative samples
        loss_pos = -mask * w_pos * torch.log(p + epsilon)
        loss_neg = -(1 - mask) * w_neg * torch.log(1 - p + epsilon)

        # Compute the total loss
        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (mask.numel() + epsilon)

        return loss


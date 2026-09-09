from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class OpenSDIMetricAccumulator:
    """OpenSDI paper metrics at the fixed 0.5 decision threshold.

    Detection F1 is accumulated from dataset-level TP/FP/FN. Localization F1
    and IoU follow IMDLBenCo 0.1.22: foreground metrics are computed per image
    and then macro-averaged over partial fake images only.
    """

    threshold: float = 0.5

    def __post_init__(self) -> None:
        self.values = torch.zeros(9, dtype=torch.float64)

    def update(
        self,
        detection_prob: torch.Tensor,
        labels: torch.Tensor,
        mask_prob: torch.Tensor,
        masks: torch.Tensor,
        localization_selector: torch.Tensor,
    ) -> None:
        detection_prob = detection_prob.detach().flatten().cpu()
        labels = labels.detach().flatten().to(torch.bool).cpu()
        pred_labels = detection_prob > float(self.threshold)
        tp = (pred_labels & labels).sum().item()
        tn = ((~pred_labels) & (~labels)).sum().item()
        fp = (pred_labels & (~labels)).sum().item()
        fn = ((~pred_labels) & labels).sum().item()
        self.values[:5] += torch.tensor([tp, tn, fp, fn, labels.numel()], dtype=torch.float64)

        selector = localization_selector.detach().flatten().to(torch.bool).cpu()
        if not selector.any():
            return
        pred = (mask_prob.detach().cpu()[selector] > float(self.threshold)).to(torch.float64)
        target = (masks.detach().cpu()[selector] > 0.5).to(torch.float64)
        pred = pred.flatten(start_dim=1)
        target = target.flatten(start_dim=1)
        intersection = (pred * target).sum(dim=1)
        pred_sum = pred.sum(dim=1)
        target_sum = target.sum(dim=1)
        # Preserve IMDLBenCo 0.1.22's exact operation order instead of using
        # the algebraically simplified Dice form. The difference is only at
        # epsilon scale, but this keeps formal paper comparison bit-for-bit in
        # line with PixelF1(mode="origin").
        precision = intersection / (pred_sum + 1.0e-8)
        recall = intersection / (target_sum + 1.0e-8)
        f1 = (2.0 * precision * recall) / (precision + recall + 1.0e-8)
        iou = intersection / (pred_sum + target_sum - intersection + 1.0e-8)
        self.values[5] += f1.sum()
        self.values[6] += iou.sum()
        self.values[7] += selector.sum().item()
        self.values[8] += target_sum.sum().item()

    def synchronize(self, device: torch.device) -> None:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        tensor = self.values.to(device)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        self.values = tensor.cpu()

    def compute(self) -> dict[str, float | int]:
        tp, tn, fp, fn, count, f1_sum, iou_sum, loc_count, foreground_pixels = self.values.tolist()
        precision = tp / (tp + fp + 1.0e-9)
        recall = tp / (tp + fn + 1.0e-9)
        detection_f1 = 2.0 * precision * recall / (precision + recall + 1.0e-9)
        return {
            "detection_f1": detection_f1,
            "detection_accuracy": (tp + tn) / max(count, 1.0),
            "localization_iou": iou_sum / max(loc_count, 1.0),
            "localization_f1": f1_sum / max(loc_count, 1.0),
            "detection_count": int(count),
            "localization_count": int(loc_count),
            "localization_foreground_pixels": int(foreground_pixels),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }

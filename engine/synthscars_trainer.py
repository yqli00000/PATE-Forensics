from __future__ import annotations

import torch

from engine.ddl_trainer import Trainer_DDL


class SynthScarsTrainer(Trainer_DDL):
    """DDL-X trainer with protocol-aligned two-class mIoU and foreground F1."""

    def __init__(self, opt):
        super().__init__(opt)
        classification_checkpoint = getattr(opt.train, "classification_checkpoint", None)
        if classification_checkpoint:
            self._load_trainable_global_classifier(str(classification_checkpoint))
        raw = getattr(getattr(opt.train, "segmentation_eval", None), "mask_thresholds", [0.5])
        self._mask_thresholds = [float(value) for value in raw]
        self._seg_confusions: list[torch.Tensor] = []

    def _load_trainable_global_classifier(self, checkpoint_path: str) -> None:
        """Restore only the coherent CLS-token classification path.

        The fused main head cannot be reused while patch/segment reducers are
        randomly reinitialized, because its input feature distribution changes.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint["state_dict"]
        prefixes = ("model.global_classifier.", "model.norm_global.")
        selected = {key[len("model."):]: value for key, value in state.items() if key.startswith(prefixes)}
        result = self.model.load_state_dict(selected, strict=False)
        expected = {f"global_classifier.{k}" for k in self.model.global_classifier.state_dict()} | {
            f"norm_global.{k}" for k in self.model.norm_global.state_dict()
        }
        loaded = set(selected)
        if not expected.issubset(loaded):
            raise RuntimeError(f"classification checkpoint is missing: {sorted(expected-loaded)}")
        print(f"[classification_init] loaded={len(selected)} trainable global CLS-path tensors from {checkpoint_path}")

    def validation_step(self, batch, batch_idx: int):
        _, metrics, outputs = self._shared_step(batch, stage="val")
        self.validation_step_outputs.append({k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in metrics.items()})
        probability = outputs["pred_mask"][:, 0]
        gt = batch["mask"][:, 0].to(probability.device) > 0.5
        per_threshold = []
        for threshold in self._mask_thresholds:
            pred = probability >= threshold
            tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum(); tn = (~pred & ~gt).sum()
            per_threshold.append(torch.stack((tp, fp, fn, tn)))
        self._seg_confusions.append(torch.stack(per_threshold).detach())
        if self._should_collect_visuals():
            self._collect_validation_visuals(batch, outputs)

    def on_validation_epoch_end(self):
        if self._seg_confusions:
            totals = torch.stack(self._seg_confusions).sum(0).to(self.device, dtype=torch.float64)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
            tp, fp, fn, tn = totals.unbind(dim=1)
            fg = tp / (tp + fp + fn).clamp_min(1)
            bg = tn / (tn + fp + fn).clamp_min(1)
            miou = (fg + bg) / 2
            f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
            best_index = int(torch.argmax(miou).item())
            fixed_index = min(range(len(self._mask_thresholds)), key=lambda i: abs(self._mask_thresholds[i] - 0.5))
            self.log("val_miou_epoch", miou[fixed_index].float(), prog_bar=False, logger=True, sync_dist=False)
            self.log("val_f1_epoch", f1[fixed_index].float(), prog_bar=False, logger=True, sync_dist=False)
            self.log("val_miou_best_epoch", miou[best_index].float(), prog_bar=True, logger=True, sync_dist=False)
            self.log("val_f1_at_best_miou_epoch", f1[best_index].float(), prog_bar=True, logger=True, sync_dist=False)
            self.log("val_best_mask_threshold", float(self._mask_thresholds[best_index]), logger=True, sync_dist=False)
            for index, threshold in enumerate(self._mask_thresholds):
                suffix = str(threshold).replace(".", "p")
                self.log(f"val_segmentation/miou_t{suffix}", miou[index].float(), logger=True, sync_dist=False)
                self.log(f"val_segmentation/f1_t{suffix}", f1[index].float(), logger=True, sync_dist=False)
            if getattr(self.trainer, "is_global_zero", False):
                self.print(
                    f"[SynthScars val] best mIoU={miou[best_index].item()*100:.2f} "
                    f"F1={f1[best_index].item()*100:.2f} threshold={self._mask_thresholds[best_index]:.2f}; "
                    f"fixed-0.5 mIoU={miou[fixed_index].item()*100:.2f} F1={f1[fixed_index].item()*100:.2f}"
                )
            self._seg_confusions.clear()
        super().on_validation_epoch_end()

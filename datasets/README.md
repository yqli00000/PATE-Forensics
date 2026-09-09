# Datasets

Datasets are not included in this repository. Prepare them separately and place
them in the directories below. All paths are relative to the repository root;

## DDL-X

```text
datasets/phase1/track1_inner/track1/train
datasets/phase1/track1_inner/track1/valid
datasets/phase2/track1/test
```

Configuration: `cfgs/train/gps_dino_mask_mixed_phase1_phase2_wo_maskloss.yaml`.

For the DDL-X inference and explanation launchers, place test images in:

```text
datasets/DDL_X_test/image/
```

## OpenSDI

```text
datasets/OpenSDI/OpenSDI_train/data/sd15-*.parquet
datasets/OpenSDI/OpenSDI_test/data/sd15-*.parquet
datasets/OpenSDI/OpenSDI_test/data/sd2-*.parquet
datasets/OpenSDI/OpenSDI_test/data/sdxl-*.parquet
datasets/OpenSDI/OpenSDI_test/data/sd3-*.parquet
datasets/OpenSDI/OpenSDI_test/data/flux-*.parquet
```

Configuration: `cfgs/train/train_opensdi.yaml` (`opensdi_datamodule.root`).

## SynthScars

```text
datasets/SynthScars/train/images/
datasets/SynthScars/train/annotations/train.json
datasets/SynthScars/test/images/
datasets/SynthScars/test/annotations/test.json
```

Configuration: `cfgs/train/train_synthscars.yaml`
(`datasets.train.root` and `datasets.val.root`). Validation is split internally
from the training data; no separate validation directory is needed.

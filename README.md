# PATE-Forensics

`PATE-Forensics` is a research codebase for image forgery detection, localization, and visual evidence description. It provides training and inference code, a reference training configuration, and a two-stage workflow: local model inference followed by optional vision-language API refinement of textual explanations.

## Contents

- `train.py`: Lightning training entrypoint.
- `infer.py`: shared checkpoint/model loading utilities used by `infer_submission.py`.
- `infer_submission.py`: local detection and localization inference with JSON export.
- `update_json_traces.py`: optional second-stage visual evidence description refinement.
- `data/`, `engine/`, `networks/`, `utils/`: source modules required for training and inference.
- `cfgs/train/gps_dino_mask_mixed_phase1_phase2_wo_maskloss.yaml`: main reference training configuration.
- `weights/`: reserved checkpoint location.
- `datasets/`: reserved dataset location.

## Environment

The project uses Python 3.10. Create and activate an environment, then install the pinned dependencies from the repository root:

```bash
conda create -n pate-forensics python=3.10 -y
conda activate pate-forensics
python -m pip install -r requirements.txt
```

The dependency versions are specified in [requirements.txt](requirements.txt), including PyTorch 2.5.1, TorchVision 0.20.1, Lightning 2.6.5, NumPy 1.26.4, Transformers 5.14.1, and the OpenAI-compatible client library 2.48.0. Use this file as the dependency reference for experiment reproduction.

Run the training and inference scripts directly from the repository root. The project does not require package installation; `pyproject.toml` contains only pytest configuration.

## Required External Files

### Weights

For training and inference, the DINOv3 backbone should be placed at:

```text
weights/dinov3-l16
```
The DINOv3 backbone version is `facebook/dinov3-vitl16-pretrain-lvd1689m`.

The DINOv3 backbone weights can be downloaded from [ModelScope](https://www.modelscope.cn/models/facebook/dinov3-vitl16-pretrain-lvd1689m).

For inference, put the checkpoint in the package-local location:

```text
weights/model_best.ckpt
```

The `model_best.ckpt` checkpoint can be downloaded from [Google Drive](https://drive.google.com/file/d/12xMXHFRo6fcOEh0vfw2yhyM5RD53nbYY/view?usp=sharing).

If the DINOv3 path saved in the checkpoint differs on the target machine, use `--backbone-path` during inference.


### Dataset

The training config expects the dataset under:

```text
datasets/phase1/track1_inner/track1/train
datasets/phase1/track1_inner/track1/valid
datasets/phase2/track1/test
```

Edit these relative paths in `cfgs/train/gps_dino_mask_mixed_phase1_phase2_wo_maskloss.yaml` if your data layout is different.

## Training

For full training reproduction, run the provided GPU training script from this folder:

```bash
bash run_gps_dino_wandb.sh
```

The script uses `cfgs/train/gps_dino_mask_mixed_phase1_phase2_wo_maskloss.yaml` and launches distributed GPU training through `torch.distributed.run`. Edit `NUM_GPUS`, `CUDA_VISIBLE_DEVICES`, W&B settings, and any resume/pretrained checkpoint arguments in the script to match the target machine.

You can also run the training entrypoint directly:

```bash
python train.py --cfg cfgs/train/gps_dino_mask_mixed_phase1_phase2_wo_maskloss.yaml --logdir gps_dino_mask_mixed_phase1_phase2_reproduce
```

To resume from a checkpoint, example command is:

```bash
python train.py --cfg cfgs/train/gps_dino_mask_mixed_phase1_phase2_wo_maskloss.yaml --resume weights/model_best.ckpt --logdir gps_dino_mask_mixed_phase1_phase2_resume
```

## Inference

Run commands from the repository root. Edit the checkpoint, input, output, and device settings in the launcher before running:

```bash
bash infer_submission.sh
```

The launcher uses an image size of 768, batch size of 4, classification threshold of 0.5, mask threshold of 0.4, and minimum component area of 8 pixels. An equivalent direct invocation is:

```bash
python infer_submission.py \
  --checkpoint weights/model_best.ckpt \
  --backbone-path weights/dinov3-l16 \
  --image-dir /path/to/images \
  --output-dir outputs/test \
  --image-size 768 --batch-size 4 --device cuda \
  --fake-threshold 0.5 --mask-threshold 0.4 --min-box-area 8 \
  --save-mask-png
```

The Python defaults differ from the launcher: image size 512, batch size 8, mask threshold 0.5, and minimum component area 16. Specify these settings explicitly when reproducing an experiment. Input images are resized without cropping; predicted masks are resized back to the original image dimensions before thresholding and connected-component extraction.

Outputs are written under:

```text
outputs/test/json/
outputs/test/mask/
outputs/test/infer_summary.json
outputs/test/infer_scores.jsonl
```

Each image JSON contains `Classification result` (`real` or `fake`), `Bounding boxes`, and `Visible forgery traces`. Boxes use inclusive `[x1, y1, x2, y2]` pixel endpoints scaled by image width/height to 0–1000. Predictions classified as real have empty boxes and zero masks. Mask PNG files are written only with `--save-mask-png`; they are needed for second-stage refinement of fake predictions.

This stage runs locally and makes no API calls. The trace field contains deterministic template text, or existing text when `--reuse-existing-traces` is enabled. Template text is not an independently generated visual explanation. API options and prompts belong exclusively to `update_json_traces.py`; the first-stage script no longer accepts `--explain-*` or `--max-api-calls`.

Use `--limit-images N` to process the first N sorted images, or `--start-index` / `--end-index` for a slice (which takes precedence over the limit). Export names use image filename stems, so input images must have unique stems even across subdirectories. Decode failures are logged and retain the legacy placeholder output (`real`, probability 0, and an error description); these placeholders must be excluded from evaluation.

The existing script filenames are retained for compatibility.

## Visual Evidence Description Refinement

Optionally refine explanations without rerunning the detection model:

```bash
export DASHSCOPE_API_KEY=your_key
bash update_submission_new.sh
```

Edit the input paths and API settings in the launcher first. The launcher selects `qwen3.5-flash`; the Python script defaults to `qwen3.6-plus`. Record the model identifier and API settings used for an experiment. Both use the configured OpenAI-compatible endpoint; an API key can also be supplied with `--explain-api-key`.

This stage reads `outputs/test/json/` and `outputs/test/mask/` and writes to `outputs/test/json_api_refined/` by default. Use `--new-json-dir` to choose a different destination; the source JSON directory cannot be used as the destination. Only `Visible forgery traces` is updated; classification, bounding boxes, and other fields are preserved.

The API is used for both real and fake predictions. Fake predictions use the original image, mask overlay, and a crop when the mask is nonempty. Real predictions are also described, including when no mask is available. A fake prediction with a missing mask is copied unchanged. API failures or empty responses retain the previous description with a classification summary appended.

`--explain-workers` controls concurrency and `--max-api-calls` limits application-level requests (the client may retry requests). `--dry-run` suppresses output JSON writes but still calls the API and writes logs and a run summary; use `--max-api-calls 0` as well for a preview without API requests.

API descriptions are conditioned on model predictions and should not be treated as ground-truth annotations. External weights and datasets are distributed separately from this source repository.

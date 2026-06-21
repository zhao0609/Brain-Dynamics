# Brain-Dynamics

Official implementation for **Brain-Dynamics: A Dual-Pipeline Framework for Visual Reconstruction from fMRI**.

Brain-Dynamics is a neuro-inspired fMRI decoding framework that combines a static global-alignment pathway with a dynamic gradient-constrained pathway. The default configuration reproduces the subject-01 experiments in the paper, and other subjects can be trained by changing `--subject`.

## Highlights

- **Dual-pipeline decoding:** a static pathway captures global fMRI-image correspondence, while a dynamic pathway constrains feature-gradient behavior associated with stimulus-sensitive neural response changes.
- **Compact CLIP target:** the encoder predicts pooled `1 x 1024` CLIP-ViT-H visual/text features, matching the feature format expected by IP-Adapter.
- **Dynamic gradient constraint:** the GD term aligns the batchwise contrastive-gradient structure between brain-predicted features and visual CLIP features.
- **IP-Adapter reconstruction:** predicted CLIP-like visual features condition frozen SDXL-Turbo + IP-Adapter for image generation, followed by the paper's candidate reranking protocol.

## Method

The brain encoder maps flattened NSD fMRI voxels to CLIP-like image and text embeddings:

```text
fMRI [B, V] -> Brain Encoder -> image feature [B, 1024], text feature [B, 1024]
```

In the paper terminology:

- **Static Pipeline:** `MSE + SoftCLIP`
- **Dynamic Pipeline:** gradient constraint / `GD`
- **Overall objective:** `MSE + SoftCLIP + GD`
- **Visual Reconstruction:** SDXL-Turbo + IP-Adapter conditioned on the predicted visual feature

The implemented objective is:

```text
L = SoftCLIP(pred_img, img_clip)
  + SoftCLIP(pred_txt, txt_clip)
  + mse_mult * (MSE(pred_img, img_clip) + MSE(pred_txt, txt_clip))
  + gd_mult * GD(pred_img, img_clip)
```

For reconstruction, `reconstruct.py` generates 16 candidate images for each test fMRI sample.

See `METHOD.md` for the paper-to-code mapping.

## Repository Structure

- `train.py`: Brain-Dynamics training.
- `reconstruct.py`: IP-Adapter reconstruction with 16-candidate reranking.
- `models.py`: CLIP wrapper and brain encoder definitions.
- `data.py`: NSD webdataset-style loader.
- `utils.py`: SoftCLIP loss, GD helper, seeding, cosine similarity.
- `ip_adapter_pipeline.py`: SDXL-Turbo pipeline wrapper for precomputed IP-Adapter embeddings.
- `nsd_access.py`: NSD caption helper.
- `METHOD.md`: implementation notes.

## Data Layout

The code expects the NSD split layout used by the experiments:

```text
{data_path}/webdataset_avg_split/train/subj01/
{data_path}/webdataset_avg_split/val/subj01/
{data_path}/webdataset_avg_split/test/subj01/
```

Each sample contains:

- `*.nsdgeneral.npy`
- `*.jpg`
- `*.coco73k.npy`

For other subjects, use the corresponding `subj02`, `subj05`, or `subj07` directories and pass `--subject`.

## Training

Subject 01:

```bash
python train.py \
  --device cuda:0 \
  --subject 1 \
  --nsd_root /data1/zhaoyuxiao/ \
  --data_path /data1/zhaoyuxiao/naturalscenesdataset/ \
  --hf_clip_path /data1/zhaoyuxiao/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/ \
  --outdir ./outputs/train_logs
```

Key defaults:

- `pool_num=15724`
- `hidden_dim=2048`
- `n_blocks=4`
- `batch_size=50`
- `lr=3e-6`
- `mse_mult=1e4`
- `gd_mult=1e8`
- `gd_tau=0.1`

Resume:

```bash
python train.py --resume /path/to/checkpoint.pth
```

## Reconstruction

```bash
python reconstruct.py \
  --device cuda:0 \
  --subject 1 \
  --ckpt ./outputs/train_logs/brain_dynamics_subj01_best_simXXXXXX.pth \
  --data_path /data1/zhaoyuxiao/naturalscenesdataset/ \
  --hf_clip_path /data1/zhaoyuxiao/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/ \
  --sdxl_turbo_path /data1/zhaoyuxiao/stabilityai/sdxl-turbo \
  --ip_adapter_path /data1/zhaoyuxiao/h94/IP-Adapter \
  --num_candidates 16 \
  --outdir ./outputs/reconstructions
```

Outputs:

- `selected/*.png`: selected reconstruction for each test sample.
- `selection_metadata.json`: selected candidate index and all candidate similarities.
- `subjXX_selected_recons.pt`: tensor stack of selected images.

Use `--save_all_candidates` to save every candidate image.

## Notes

- This repository does not include NSD data, COCO annotations, CLIP weights, SDXL-Turbo weights, IP-Adapter weights, or trained checkpoints.

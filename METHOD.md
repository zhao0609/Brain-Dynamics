# Method Notes

This document maps the Brain-Dynamics manuscript terminology to the released implementation.

## Brain Encoder

The encoder is an MLP-style residual network:

```text
fMRI voxel vector
  -> subject-specific adapter
  -> Linear + LayerNorm + GELU + Dropout
  -> ResMLP blocks
  -> image head [B, 1024]
  -> text head  [B, 1024]
```

The default experiment uses subject 01. The script is parameterized by `--subject`.

## Static Pipeline

The static pathway learns global alignment between fMRI-derived features and frozen CLIP targets:

```text
L_static =
    SoftCLIP(pred_image, clip_image)
  + SoftCLIP(pred_text, clip_text)
  + mse_mult * MSE(pred_image, clip_image)
  + mse_mult * MSE(pred_text, clip_text)
```

The CLIP targets are pooled CLIP-ViT-H embeddings:

```text
clip_image: [B, 1024]
clip_text:  [B, 1024]
```

## Dynamic Pipeline

The dynamic pathway is the gradient constraint. It encourages the predicted brain feature space to exhibit a similar batchwise variation structure to the target visual feature space:

```text
t_grad = get_grad(clip_image, clip_image)
s_grad = get_grad(pred_image, pred_image)
L_dynamic = MSE(s_grad, stopgrad(t_grad))
```

The helper computes softmax-gradient-like vectors:

```text
logits = p @ k.T / tau
prob = softmax(logits)
grad_p = prob @ k / tau / batch_size
```

This is the implemented dynamic constraint used by the code.

## Overall Objective

```text
L_total = L_static + gd_mult * L_dynamic
```

The default hyperparameters are:

```text
mse_mult = 1e4
gd_mult  = 1e8
gd_tau   = 0.1
```

## Visual Reconstruction

After training, the predicted `1 x 1024` visual feature conditions the frozen IP-Adapter/SDXL generator:

```text
pred_image = BrainEncoder(fMRI)
candidate_j = IPAdapter_SDXL(pred_image), j = 1...16
```

The final reconstruction is selected with the candidate reranking protocol:

```text
argmax_j cosine(CLIP_image(candidate_j), CLIP_image(original_stimulus))
```

This selection is performed only after candidates are generated. The original stimulus does not condition generation.

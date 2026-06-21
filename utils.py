import os
import random

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed=42, cudnn_deterministic=True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True


def check_loss(loss, message="loss"):
    if torch.isnan(loss).any():
        raise ValueError(f"NaN loss in {message}")


def soft_clip_loss(preds, targs, temp=0.005, eps=1e-10):
    """Soft-label contrastive loss used in the original experiment.

    The target-target similarity matrix defines soft labels, and the
    brain-predicted-to-target similarity matrix is trained to match them in
    both directions.
    """
    clip_clip = (targs @ targs.T) / temp + eps
    brain_clip = (preds @ targs.T) / temp + eps

    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    return (loss1 + loss2) / 2


def get_grad(p, k, tau=0.1):
    """Gradient-matching term used by the 1x1024 GD experiment.

    This is the simplified GD variant used in the experiment code: compute
    batchwise softmax similarities and match the resulting gradient-like
    vectors between real CLIP image embeddings and brain-predicted embeddings.
    """
    logits = p @ k.T / tau
    prob = F.softmax(logits, dim=1)
    grad_p = prob @ k / tau / prob.size(0)

    embed_size = p.size(1)
    prob_repeat = prob.t().repeat(1, embed_size).view(-1, embed_size, p.size(0))
    grad_k = (prob_repeat * (p.t() / tau).unsqueeze(0)).sum(-1) / prob.size(0)
    return grad_p, grad_k


def batchwise_cosine_similarity(query, candidates, eps=1e-8):
    """Return cosine similarity of each candidate against one or more queries."""
    query = F.normalize(query, dim=-1, eps=eps)
    candidates = F.normalize(candidates, dim=-1, eps=eps)
    return query @ candidates.T

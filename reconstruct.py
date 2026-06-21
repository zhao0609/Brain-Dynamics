import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

import data
import utils
from ip_adapter_pipeline import IPAdapterGenerator
from models import Clipper, MindSingle


def pil_to_tensor(image):
    image = image.convert("RGB")
    arr = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


@torch.no_grad()
def load_voxel2clip(args):
    model = MindSingle(
        in_dim=args.pool_num,
        out_dim_image=1024,
        out_dim_text=1024,
        h=args.hidden_dim,
        n_blocks=args.n_blocks,
        subj_list=[args.subject],
    ).to(args.device)
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    state_dict = checkpoint["voxel2clip"] if "voxel2clip" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval().requires_grad_(False)
    return model


@torch.no_grad()
def original_stimulus_embedding(clip_extractor, image):
    _, pooled = clip_extractor.embed_image(image)
    return nn.functional.normalize(pooled.view(len(pooled), -1), dim=-1)


@torch.no_grad()
def reconstruction_embedding(clip_extractor, pil_image):
    image_tensor = pil_to_tensor(pil_image)
    _, pooled = clip_extractor.embed_image(image_tensor)
    return nn.functional.normalize(pooled.view(len(pooled), -1), dim=-1)


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct images with Brain-Dynamics and IP-Adapter."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=99999)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--data_path", default="/data1/zhaoyuxiao/naturalscenesdataset/")
    parser.add_argument("--hf_clip_path", default="/data1/zhaoyuxiao/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/")
    parser.add_argument("--ckpt", required=True, help="Brain-Dynamics checkpoint produced by train.py")
    parser.add_argument("--outdir", default="./outputs/reconstructions")
    parser.add_argument("--sdxl_turbo_path", default="/data1/zhaoyuxiao/stabilityai/sdxl-turbo")
    parser.add_argument("--ip_adapter_path", default="/data1/zhaoyuxiao/h94/IP-Adapter")
    parser.add_argument("--ip_adapter_subfolder", default="sdxl_models")
    parser.add_argument("--ip_adapter_weight", default="ip-adapter_sdxl_vit-h.safetensors")
    parser.add_argument("--ip_adapter_scale", type=float, default=1.0)
    parser.add_argument("--num_candidates", type=int, default=16)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--pool_num", type=int, default=15724)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--save_all_candidates", action="store_true")
    args = parser.parse_args()

    utils.seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    selected_dir = os.path.join(args.outdir, "selected")
    candidate_dir = os.path.join(args.outdir, "candidates")
    os.makedirs(selected_dir, exist_ok=True)
    if args.save_all_candidates:
        os.makedirs(candidate_dir, exist_ok=True)

    test_dl = data.get_subject_test_dl(
        subject=args.subject,
        data_path=args.data_path.rstrip("/"),
        batch_size=1,
        num_workers=0,
        pool_type="max",
        pool_num=args.pool_num,
        seed=args.seed,
    )
    voxel2clip = load_voxel2clip(args)
    clip_extractor = Clipper(
        "ViT-L/14",
        device=args.device,
        hidden_state=True,
        norm_embs=True,
        hf_clip_path=args.hf_clip_path,
    )
    generator = IPAdapterGenerator(
        sdxl_turbo_path=args.sdxl_turbo_path,
        ip_adapter_path=args.ip_adapter_path,
        ip_adapter_subfolder=args.ip_adapter_subfolder,
        ip_adapter_weight=args.ip_adapter_weight,
        num_inference_steps=args.num_inference_steps,
        device=args.device,
        ip_adapter_scale=args.ip_adapter_scale,
    )

    metadata = {
        "subject": f"subj{args.subject:02d}",
        "embedding_target": "1x1024 CLIP pooled image embedding",
        "candidate_count": args.num_candidates,
        "selection_target": "original stimulus image CLIP pooled embedding",
        "selection_metric": "cosine_similarity",
        "items": [],
    }
    selected_tensors = []

    for sample_idx, batch in enumerate(test_dl):
        if sample_idx < args.start:
            continue
        if args.end is not None and sample_idx >= args.end:
            break

        voxel, image, coco, _ = batch
        voxel = torch.mean(voxel, axis=1).float().to(args.device)
        image = image.to(args.device)

        clip_image_pred, _ = voxel2clip(voxel)
        brain_image_embed = nn.functional.normalize(clip_image_pred, dim=-1)
        stim_embed = original_stimulus_embedding(clip_extractor, image)

        best_score = -float("inf")
        best_candidate_idx = -1
        best_image = None
        best_tensor = None
        scores = []

        for candidate_idx in range(args.num_candidates):
            torch_generator = torch.Generator(device=args.device)
            torch_generator.manual_seed(args.seed + sample_idx * args.num_candidates + candidate_idx)
            candidate = generator.generate(brain_image_embed.to(dtype=torch.float16), generator=torch_generator)
            candidate_embed = reconstruction_embedding(clip_extractor, candidate).to(stim_embed.device)
            score = utils.batchwise_cosine_similarity(stim_embed, candidate_embed)[0, 0].item()
            scores.append(score)

            if args.save_all_candidates:
                cand_path = os.path.join(candidate_dir, f"{sample_idx:05d}_{candidate_idx:02d}_sim{score:.6f}.png")
                candidate.save(cand_path)

            if score > best_score:
                best_score = score
                best_candidate_idx = candidate_idx
                best_image = candidate
                best_tensor = pil_to_tensor(candidate)

        selected_path = os.path.join(selected_dir, f"{sample_idx:05d}_pick{best_candidate_idx:02d}_sim{best_score:.6f}.png")
        best_image.save(selected_path)
        selected_tensors.append(best_tensor)

        metadata["items"].append(
            {
                "sample_index": sample_idx,
                "coco73k_id": int(coco.squeeze().item()),
                "selected_candidate": best_candidate_idx,
                "selected_similarity": best_score,
                "all_candidate_similarities": scores,
                "selected_path": selected_path,
            }
        )
        print(f">>> sample {sample_idx}: picked {best_candidate_idx}/{args.num_candidates} sim={best_score:.6f}", flush=True)

    if selected_tensors:
        torch.save(torch.cat(selected_tensors, dim=0), os.path.join(args.outdir, f"subj{args.subject:02d}_selected_recons.pt"))
    with open(os.path.join(args.outdir, "selection_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved reconstructions to {args.outdir}")


if __name__ == "__main__":
    main()

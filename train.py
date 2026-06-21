import argparse
import os
import time

import numpy as np
import torch
from nsd_access import NSDAccess
from torch import nn

import data
import utils
from models import Clipper, MindSingle


def prepare_coco(nsd_root):
    nsda = NSDAccess(nsd_root)
    coco_73k = list(range(73000))
    prompts_list = nsda.read_image_coco_info(coco_73k, info_type="captions")
    print("COCO captions loaded.")
    return prompts_list


def prepare_clip(args):
    clip_extractor = Clipper(
        args.clip_variant,
        device=args.device,
        hidden_state=True,
        norm_embs=True,
        hf_clip_path=args.hf_clip_path,
    )
    args.clip_extractor = clip_extractor
    return 1024, 1024


def prepare_model(args, out_dim_image, out_dim_text):
    voxel2clip = MindSingle(
        in_dim=args.pool_num,
        out_dim_image=out_dim_image,
        out_dim_text=out_dim_text,
        h=args.hidden_dim,
        n_blocks=args.n_blocks,
        subj_list=[args.subject],
    ).to(args.device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        state_dict = checkpoint["voxel2clip"] if "voxel2clip" in checkpoint else checkpoint
        voxel2clip.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {args.resume}")

    no_decay = ["bias", "Norm", "temperature"]
    opt_grouped_parameters = [
        {
            "params": [p for n, p in voxel2clip.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in voxel2clip.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    args.optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=args.lr)
    args.scheduler = torch.optim.lr_scheduler.LinearLR(args.optimizer, total_iters=args.num_epochs, last_epoch=-1)
    args.voxel2clip = voxel2clip


def get_captions(coco_tensor, prompts_list, repeat_index):
    coco_ids = coco_tensor.squeeze().tolist()
    if isinstance(coco_ids, int):
        coco_ids = [coco_ids]
    current_prompts = [prompts_list[coco_id] for coco_id in coco_ids]
    return [prompts[repeat_index]["caption"] for prompts in current_prompts]


def train_step(args, voxel, image, captions):
    args.optimizer.zero_grad()
    voxel = voxel.to(args.device)
    image = data.img_augment(image)

    _, clip_image_pooled = args.clip_extractor.embed_image(image)
    _, clip_text_pooled = args.clip_extractor.embed_text(captions)

    clip_image_pred, clip_text_pred = args.voxel2clip(voxel)
    clip_image_pred_norm = nn.functional.normalize(clip_image_pred, dim=-1)
    clip_text_pred_norm = nn.functional.normalize(clip_text_pred, dim=-1)
    clip_image_norm = nn.functional.normalize(clip_image_pooled, dim=-1)
    clip_text_norm = nn.functional.normalize(clip_text_pooled, dim=-1)

    loss_clip_image = utils.soft_clip_loss(clip_image_pred_norm, clip_image_norm)
    loss_clip_text = utils.soft_clip_loss(clip_text_pred_norm, clip_text_norm)
    loss_mse_image = nn.MSELoss()(clip_image_pred_norm, clip_image_norm)
    loss_mse_text = nn.MSELoss()(clip_text_pred_norm, clip_text_norm)

    t_grad_p_img, _ = utils.get_grad(clip_image_norm, clip_image_norm, tau=args.gd_tau)
    s_grad_p_img, _ = utils.get_grad(clip_image_pred_norm, clip_image_pred_norm, tau=args.gd_tau)
    loss_gd = nn.functional.mse_loss(s_grad_p_img, t_grad_p_img.detach())

    loss = (
        loss_clip_image
        + loss_clip_text
        + args.mse_mult * (loss_mse_image + loss_mse_text)
        + args.gd_mult * loss_gd
    )
    utils.check_loss(loss)
    loss.backward()
    args.optimizer.step()
    args.scheduler.step()

    return {
        "loss": loss.item(),
        "clip_image": loss_clip_image.item(),
        "clip_text": loss_clip_text.item(),
        "mse_image": loss_mse_image.item(),
        "mse_text": loss_mse_text.item(),
        "gd": loss_gd.item(),
    }


@torch.no_grad()
def eval_epoch(args, epoch, val_dl, prompts_list):
    args.voxel2clip.eval()
    val_sims_image = 0.0
    val_sims_text = 0.0
    n_batches = 0
    for val_i, batch in enumerate(val_dl):
        repeat_index = val_i % 3
        voxel, image, coco, _ = batch
        voxel = torch.mean(voxel, axis=1).float().to(args.device)
        captions = get_captions(coco, prompts_list, repeat_index)

        _, clip_image_pooled = args.clip_extractor.embed_image(image)
        _, clip_text_pooled = args.clip_extractor.embed_text(captions)
        clip_image_pred, clip_text_pred = args.voxel2clip(voxel)

        clip_image_pred_norm = nn.functional.normalize(clip_image_pred, dim=-1)
        clip_text_pred_norm = nn.functional.normalize(clip_text_pred, dim=-1)
        clip_image_norm = nn.functional.normalize(clip_image_pooled, dim=-1)
        clip_text_norm = nn.functional.normalize(clip_text_pooled, dim=-1)

        val_sims_image += nn.functional.cosine_similarity(clip_image_norm, clip_image_pred_norm).mean().item()
        val_sims_text += nn.functional.cosine_similarity(clip_text_norm, clip_text_pred_norm).mean().item()
        n_batches += 1

    current_sim = (val_sims_image + val_sims_text) / max(n_batches, 1)
    print(f">>> Epoch {epoch} eval similarity: {current_sim:.6f}")
    return current_sim


def save_checkpoint(args, epoch, tag):
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"brain_dynamics_subj{args.subject:02d}_{tag}.pth")
    torch.save({"epoch": epoch, "voxel2clip": args.voxel2clip.state_dict()}, path)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Train Brain-Dynamics fMRI-to-CLIP mapping.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--nsd_root", default="/data1/zhaoyuxiao/")
    parser.add_argument("--data_path", default="/data1/zhaoyuxiao/naturalscenesdataset/")
    parser.add_argument("--hf_clip_path", default="/data1/zhaoyuxiao/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/")
    parser.add_argument("--outdir", default="./outputs/train_logs")
    parser.add_argument("--resume", default="")
    parser.add_argument("--clip_variant", default="ViT-L/14")
    parser.add_argument("--num_epochs", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--val_batch_size", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pool_num", type=int, default=15724)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--mse_mult", type=float, default=1e4)
    parser.add_argument("--gd_mult", type=float, default=1e8)
    parser.add_argument("--gd_tau", type=float, default=0.1)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=10)
    args = parser.parse_args()

    utils.seed_everything(args.seed)
    prompts_list = prepare_coco(args.nsd_root)
    train_dl, val_dl = data.get_subject_dls(
        subject=args.subject,
        data_path=args.data_path.rstrip("/"),
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        pool_type="max",
        pool_num=args.pool_num,
        length=None,
        seed=args.seed,
    )
    out_dim_image, out_dim_text = prepare_clip(args)
    prepare_model(args, out_dim_image, out_dim_text)

    best_sim = -float("inf")
    start_time = time.time()
    for epoch in range(args.num_epochs):
        args.voxel2clip.train()
        for train_i, batch in enumerate(train_dl):
            repeat_index = train_i % 3
            voxel, image, coco, _ = batch
            voxel = voxel[:, repeat_index, ...].float()
            captions = get_captions(coco, prompts_list, repeat_index)
            metrics = train_step(args, voxel, image, captions)
            print(
                f">>> Epoch {epoch} Iter {train_i} "
                f"loss={metrics['loss']:.6f} soft_img={metrics['clip_image']:.6f} "
                f"soft_txt={metrics['clip_text']:.6f} mse_img={metrics['mse_image']:.8f} "
                f"mse_txt={metrics['mse_text']:.8f} gd={metrics['gd']:.10f}",
                flush=True,
            )

        if epoch % args.eval_interval == 0:
            current_sim = eval_epoch(args, epoch, val_dl, prompts_list)
            if current_sim > best_sim:
                best_sim = current_sim
                save_checkpoint(args, epoch, f"best_sim{current_sim:.6f}")

        if epoch % args.save_interval == 0:
            save_checkpoint(args, epoch, f"last_{epoch}")

    print(f"Finished in {(time.time() - start_time) / 3600:.2f} hours")


if __name__ == "__main__":
    main()

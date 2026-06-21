import math
import random

import clip
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection


class Clipper(torch.nn.Module):
    """Frozen CLIP wrapper used for 1x1024 target embeddings."""

    def __init__(
        self,
        clip_variant,
        clamp_embs=False,
        norm_embs=False,
        hidden_state=False,
        device=torch.device("cpu"),
        hf_clip_path="/data1/zhaoyuxiao/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/",
    ):
        super().__init__()
        assert clip_variant in ("RN50", "ViT-L/14", "ViT-B/32", "RN50x64")
        self.hidden_state = hidden_state
        self.clamp_embs = clamp_embs
        self.norm_embs = norm_embs
        self.device = device

        if clip_variant == "ViT-L/14" and hidden_state:
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(hf_clip_path).eval().to(device)
            text_encoder = CLIPTextModelWithProjection.from_pretrained(hf_clip_path).eval().to(device)
            for param in image_encoder.parameters():
                param.requires_grad = False
            for param in text_encoder.parameters():
                param.requires_grad = False
            self.image_encoder = image_encoder
            self.text_encoder = text_encoder
            self.tokenizer = CLIPTokenizer.from_pretrained(hf_clip_path)
        elif hidden_state:
            raise ValueError("hidden_state=True is only supported for ViT-L/14 in this code.")

        clip_model, _ = clip.load(clip_variant, device=device)
        clip_model.eval()
        for param in clip_model.parameters():
            param.requires_grad = False
        self.clip = clip_model
        self.clip_variant = clip_variant
        self.clip_size = (448, 448) if clip_variant == "RN50x64" else (224, 224)

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(size=self.clip_size[0], interpolation=transforms.InterpolationMode.BICUBIC, antialias=None),
                transforms.CenterCrop(size=self.clip_size),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def embed_image(self, image):
        """Return token-level image embeddings and pooled 1x1024 image embeddings."""
        if self.hidden_state:
            clip_input = self.preprocess(image.to(self.device))
            encoder_output = self.image_encoder(clip_input)
            token_embeds = encoder_output.last_hidden_state
            pooled_embeds = encoder_output.image_embeds
            token_embeds = self.image_encoder.vision_model.post_layernorm(token_embeds)
            token_embeds = self.image_encoder.visual_projection(token_embeds)
        else:
            token_embeds = self.preprocess(image.to(self.device))
            token_embeds = self.clip.encode_image(token_embeds)
            pooled_embeds = token_embeds

        if self.clamp_embs:
            token_embeds = torch.clamp(token_embeds, -1.5, 1.5)
        if self.norm_embs:
            if self.hidden_state:
                token_embeds = token_embeds / torch.norm(token_embeds[:, 0], dim=-1).reshape(-1, 1, 1)
            else:
                token_embeds = nn.functional.normalize(token_embeds, dim=-1)
        return token_embeds, pooled_embeds

    def embed_text(self, prompt):
        """Return token-level text embeddings and pooled 1x1024 text embeddings."""
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            encoder_output = self.text_encoder(text_inputs.input_ids.to(self.device))

        token_embeds = self.text_encoder.text_projection(encoder_output.last_hidden_state)
        pooled_embeds = encoder_output.text_embeds
        token_embeds = token_embeds / torch.norm(pooled_embeds.unsqueeze(1), dim=-1, keepdim=True)
        return token_embeds, pooled_embeds


class AdapterLayer(nn.Module):
    def __init__(self, in_channels, bottleneck=128, out_channels=None, dropout=0.0, adapter_scalar="1.0"):
        super().__init__()
        self.in_channels = in_channels
        self.down_size = bottleneck
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.scale = nn.Parameter(torch.ones(1)) if adapter_scalar == "learnable_scalar" else float(adapter_scalar)
        self.down_proj = nn.Linear(self.in_channels, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.out_channels)
        self.dropout = dropout
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        residual = x
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        return self.up_proj(down) * self.scale + residual


class ResMLP(nn.Module):
    def __init__(self, h, n_blocks, dropout=0.15):
        super().__init__()
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(h, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x):
        residual = x
        for block in self.mlp:
            x = block(x)
            x += residual
            residual = x
        return x


class MindSingle(nn.Module):
    """Single-subject fMRI-to-CLIP encoder used by Brain-Dynamics."""

    def __init__(self, in_dim=15724, out_dim_image=1024, out_dim_text=1024, h=2048, n_blocks=4, subj_list=None):
        super().__init__()
        subj_list = subj_list or [1]
        self.subj_list = subj_list
        self.embedder = nn.ModuleDict(
            {
                str(subj): nn.Sequential(
                    AdapterLayer(in_dim, 128),
                    nn.Linear(in_dim, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(0.5),
                )
                for subj in subj_list
            }
        )
        self.translator = ResMLP(h, n_blocks)
        self.head_image = nn.Linear(h, out_dim_image)
        self.head_text = nn.Linear(h, out_dim_text)

    def forward(self, x):
        x = self.embedder[str(self.subj_list[0])](x)
        x = self.translator(x)
        return self.head_image(x).reshape(len(x), -1), self.head_text(x).reshape(len(x), -1)


class MindBridge(MindSingle):
    """Reference multi-subject bridge module kept for compatibility."""

    def __init__(self, in_dim=15724, out_dim_image=1024, out_dim_text=1024, h=2048, n_blocks=4, subj_list=None, adapting=False):
        assert subj_list is not None and len(subj_list) >= 2, "MindBridge requires at least two subjects"
        super().__init__(in_dim=in_dim, out_dim_image=out_dim_image, out_dim_text=out_dim_text, h=h, n_blocks=n_blocks, subj_list=subj_list)
        self.builder = nn.ModuleDict(
            {
                str(subj): nn.Sequential(
                    nn.Linear(h, in_dim),
                    nn.LayerNorm(in_dim),
                    nn.GELU(),
                    AdapterLayer(in_dim, 128),
                )
                for subj in subj_list
            }
        )
        self.adapting = adapting
        self.cyc_loss = nn.MSELoss()

    def forward(self, x):
        if len(x) == 2 and type(x) is tuple:
            subj_list = x[1].tolist()
            x = x[0]
        else:
            subj_list = self.subj_list

        x = x.squeeze()
        x_subj = torch.chunk(x, len(subj_list))
        encoded = []
        rec = []
        subj_a, subj_b = (subj_list[0], subj_list[-1]) if self.adapting else random.sample(subj_list, 2)
        for idx, subj_i in enumerate(subj_list):
            x_i = self.embedder[str(subj_i)](x_subj[idx])
            if subj_i == subj_a:
                x_a = x_i
            encoded.append(x_i)
            rec.append(self.builder[str(subj_i)](x_i))

        x_b = self.builder[str(subj_b)](x_a)
        x_b = self.embedder[str(subj_b)](x_b)
        loss_cyc = self.cyc_loss(x_a, x_b)
        x = self.translator(torch.cat(encoded, dim=0))
        return self.head_image(x).reshape(len(x), -1), self.head_text(x).reshape(len(x), -1), torch.cat(rec, dim=0), loss_cyc

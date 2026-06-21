import torch
from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipelineOutput,
    rescale_noise_cfg,
    retrieve_timesteps,
)


@torch.no_grad()
def generate_ip_adapter_embeds(
    self,
    prompt="",
    prompt_2=None,
    height=None,
    width=None,
    num_inference_steps=4,
    timesteps=None,
    denoising_end=None,
    guidance_scale=0.0,
    negative_prompt=None,
    negative_prompt_2=None,
    num_images_per_prompt=1,
    eta=0.0,
    generator=None,
    latents=None,
    prompt_embeds=None,
    negative_prompt_embeds=None,
    pooled_prompt_embeds=None,
    negative_pooled_prompt_embeds=None,
    ip_adapter_embeds=None,
    output_type="pil",
    return_dict=True,
    cross_attention_kwargs=None,
    guidance_rescale=0.0,
    original_size=None,
    crops_coords_top_left=(0, 0),
    target_size=None,
    negative_original_size=None,
    negative_crops_coords_top_left=(0, 0),
    negative_target_size=None,
    clip_skip=None,
    callback_on_step_end=None,
    callback_on_step_end_tensor_inputs=("latents",),
):
    """SDXL call path that accepts precomputed 1x1024 IP-Adapter embeddings.

    This is a compact version of the experiment pipeline: the only custom
    behavior is passing `ip_adapter_embeds` directly as added image conditions.
    """
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor
    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        None,
        negative_prompt,
        negative_prompt_2,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        list(callback_on_step_end_tensor_inputs),
    )

    self._guidance_scale = guidance_scale
    self._guidance_rescale = guidance_rescale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs
    self._denoising_end = denoising_end

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    lora_scale = self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        lora_scale=lora_scale,
        clip_skip=self.clip_skip,
    )

    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        self.unet.config.in_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    add_text_embeds = pooled_prompt_embeds
    text_encoder_projection_dim = (
        int(pooled_prompt_embeds.shape[-1])
        if self.text_encoder_2 is None
        else self.text_encoder_2.config.projection_dim
    )
    add_time_ids = self._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = self._get_add_time_ids(
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    else:
        negative_add_time_ids = add_time_ids

    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

    image_embeds = None
    if ip_adapter_embeds is not None:
        image_embeds = ip_adapter_embeds.to(device=device, dtype=prompt_embeds.dtype)
        if self.do_classifier_free_guidance:
            image_embeds = torch.cat([torch.zeros_like(image_embeds), image_embeds], dim=0)

    if self.denoising_end is not None and isinstance(self.denoising_end, float) and 0 < self.denoising_end < 1:
        cutoff = int(round(self.scheduler.config.num_train_timesteps - (self.denoising_end * self.scheduler.config.num_train_timesteps)))
        num_inference_steps = len([t for t in timesteps if t >= cutoff])
        timesteps = timesteps[:num_inference_steps]

    timestep_cond = None
    if self.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = self.get_guidance_scale_embedding(
            guidance_scale_tensor,
            embedding_dim=self.unet.config.time_cond_proj_dim,
        ).to(device=device, dtype=latents.dtype)

    self._num_timesteps = len(timesteps)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, timestep in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            if image_embeds is not None:
                added_cond_kwargs["image_embeds"] = image_embeds

            noise_pred = self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

            latents = self.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False)[0]
            if callback_on_step_end is not None:
                callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                callback_outputs = callback_on_step_end(self, i, timestep, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                negative_pooled_prompt_embeds = callback_outputs.pop("negative_pooled_prompt_embeds", negative_pooled_prompt_embeds)
                add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

            if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                progress_bar.update()

    if output_type == "latent":
        image = latents
    else:
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        if self.watermark is not None:
            image = self.watermark.apply_watermark(image)
        image = self.image_processor.postprocess(image, output_type=output_type)

    self.maybe_free_model_hooks()
    if not return_dict:
        return (image,)
    return StableDiffusionXLPipelineOutput(images=image)


class IPAdapterGenerator:
    def __init__(
        self,
        sdxl_turbo_path="stabilityai/sdxl-turbo",
        ip_adapter_path="h94/IP-Adapter",
        ip_adapter_subfolder="sdxl_models",
        ip_adapter_weight="ip-adapter_sdxl_vit-h.safetensors",
        num_inference_steps=4,
        device="cuda:0",
        ip_adapter_scale=1.0,
    ):
        self.num_inference_steps = num_inference_steps
        self.device = device
        self.dtype = torch.float16

        pipe = DiffusionPipeline.from_pretrained(
            sdxl_turbo_path,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe.to(device)
        pipe.generate_ip_adapter_embeds = generate_ip_adapter_embeds.__get__(pipe)
        pipe.load_ip_adapter(
            ip_adapter_path,
            subfolder=ip_adapter_subfolder,
            weight_name=ip_adapter_weight,
            torch_dtype=torch.float16,
        )
        pipe.set_ip_adapter_scale(ip_adapter_scale)
        self.pipe = pipe

    @torch.no_grad()
    def generate(self, image_embeds, generator=None):
        image_embeds = image_embeds.to(device=self.device, dtype=self.dtype)
        return self.pipe.generate_ip_adapter_embeds(
            prompt="",
            ip_adapter_embeds=image_embeds,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]

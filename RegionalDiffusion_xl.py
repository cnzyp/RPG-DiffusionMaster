# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import numpy as np
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from cross_attention import hook_forwards,TOKENS,TOKENSCON
from matrix import matrixdealer,keyconverter
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    FusedAttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers,DPMSolverMultistepScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_invisible_watermark_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput

# Multi-Adapter support
from multi_ip_adapter_mixin import MultiIPAdapterMixin
from spatial_aware_attention import (
    SpatialAwareIPAdapter,
    register_spatial_aware_hooks,
    remove_spatial_aware_hooks,
    compute_latent_size_from_resolution
)


if is_invisible_watermark_available():
    from .watermark import StableDiffusionXLWatermarker

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionXLPipeline

        >>> pipe = StableDiffusionXLPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class RegionalDiffusionXLPipeline(
    MultiIPAdapterMixin,
    DiffusionPipeline,
    StableDiffusionMixin,
    FromSingleFileMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
    IPAdapterMixin,
):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion XL.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion XL uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([` CLIPTextModelWithProjection`]):
            Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
            Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
        add_watermarker (`bool`, *optional*):
            Whether to use the [invisible_watermark library](https://github.com/ShieldMnt/invisible-watermark/) to
            watermark output images. If not defined, it will default to True if the package is installed, otherwise no
            watermarker will be used.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "feature_extractor",
    ]
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
        "negative_pooled_prompt_embeds",
        "negative_add_time_ids",
    ]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DPMSolverMultistepScheduler,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        hook_forwards(self, self.unet)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.default_sample_size = self.unet.config.sample_size

        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None

        # IP-Adapter FaceID components (initialized when load_ip_adapter_faceid is called)
        self.faceid_adapter = None
        self.category_detector = None
        self._faceid_loaded = False
        self._ip_adapter_scale = 0.7  # 保存IP-Adapter Plus的scale

        # Multi-Adapter support
        self._init_multi_adapter()
        self.spatial_aware_adapter = None
        self.spatial_aware_hooks = []

    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder, lora_scale)

            if self.text_encoder_2 is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # textual inversion: process multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)
                    
                # split prompt into regional prompts
                regional_prompt_list = prompt[0].split('BREAK')
                self._regional_split_prompts = [p.strip() for p in regional_prompt_list]
                regional_prompt_embeds = []
                
                for sub_prompt in regional_prompt_list:
                    text_inputs = tokenizer(
                        sub_prompt,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )

                    text_input_ids = text_inputs.input_ids
                    untruncated_ids = tokenizer(sub_prompt, padding="longest", return_tensors="pt").input_ids

                    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                        text_input_ids, untruncated_ids
                    ):
                        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                        logger.warning(
                            "The following part of your input was truncated because CLIP can only handle sequences up to"
                            f" {tokenizer.model_max_length} tokens: {removed_text}"
                        )

                    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

                    # We are only ALWAYS interested in the pooled output of the final text encoder
                    pooled_prompt_embeds = prompt_embeds[0]
                    if clip_skip is None:
                        prompt_embeds = prompt_embeds.hidden_states[-2]
                    else:
                        # "2" because SDXL always indexes from the penultimate layer.
                        prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]
                    # print('sub_prompt_embeds:',prompt_embeds.shape)
                    regional_prompt_embeds.append(prompt_embeds)
                prompt_embeds = torch.cat(regional_prompt_embeds, dim=1)
                #print('regional_concatenated_prompt_embeds:',prompt_embeds.shape)
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

            # Identity in the text stream.  RPG already routes each region's
            # tokens to that region's pixels, so fusing subject i's identity
            # into region i's trigger token makes the identity regional without
            # any mask: nothing is injected spatially, so nothing competes with
            # the prompt for framing, pose or scene.
            _pm = getattr(self, "_regional_id_fuser", None)
            if _pm is not None and getattr(self, "_regional_split_prompts", None):
                prompt_embeds = _pm(prompt_embeds, self._regional_split_prompts)
            region_num = prompt_embeds.shape[1] // TOKENSCON
            print(region_num)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)
                regional_negative_prompt_list = []
                for i in range(region_num):
                    max_length = TOKENSCON
                    uncond_input = tokenizer(
                        negative_prompt,
                        padding="max_length",
                        max_length=max_length,
                        truncation=True,
                        return_tensors="pt",
                    )

                    negative_prompt_embeds = text_encoder(
                        uncond_input.input_ids.to(device),
                        output_hidden_states=True,
                    )
                    # We are only ALWAYS interested in the pooled output of the final text encoder
                    negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                    negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
                    
                    
                    #print('negative_prompt_embeds:',negative_prompt_embeds.shape)
                    regional_negative_prompt_list.append(negative_prompt_embeds)
                negative_prompt_embeds = torch.concat(regional_negative_prompt_list,dim=1)
                #print('negative_prompt_embeds',negative_prompt_embeds.shape)
                negative_prompt_embeds_list.append(negative_prompt_embeds)
                

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)
            #print('negative_prompt_embeds',negative_prompt_embeds.shape)
        if self.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        else:
            prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            if self.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.unet.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds

    def load_ip_adapter_faceid(
        self,
        pretrained_model_name_or_path: str,
        weight_name: str = "ip-adapter-faceid-plusv2_sdxl.bin",
        lora_weight_name: str = "ip-adapter-faceid-plusv2_sdxl_lora.safetensors",
        device: str = "cuda",
        **kwargs
    ):
        """
        Load IP-Adapter FaceID model for precise face identity preservation.

        Args:
            pretrained_model_name_or_path: Path to FaceID model directory
            weight_name: FaceID adapter weights filename
            lora_weight_name: LoRA weights filename
            device: Device for FaceID adapter ('cuda' or 'cpu')
        """
        from faceid_adapter import FaceIDAdapter
        from category_detector import CategoryDetector
        from pathlib import Path
        import safetensors.torch

        print(f"[load_ip_adapter_faceid] Loading FaceID model from {pretrained_model_name_or_path}")

        # 1. Load FaceID adapter weights
        model_path = Path(pretrained_model_name_or_path)
        faceid_weight_path = model_path / weight_name

        if not faceid_weight_path.exists():
            raise ValueError(f"FaceID weight not found: {faceid_weight_path}")

        print(f"  Loading FaceID weights: {faceid_weight_path.name}")
        faceid_state_dict = torch.load(faceid_weight_path, map_location="cpu")

        # 2. Create FaceID projection layer (512 -> cross_attention_dim)
        # FaceID uses 512-dim face embeddings instead of 1280-dim CLIP embeddings
        cross_attention_dim = self.unet.config.cross_attention_dim
        print(f"  Creating FaceID projection layer: 512 -> {cross_attention_dim}")

        # Simple linear projection for FaceID embeddings
        self.unet.encoder_hid_proj_faceid = torch.nn.Linear(512, cross_attention_dim).to(self.unet.device, dtype=self.unet.dtype)

        # Load projection weights from state_dict if available
        if 'proj' in faceid_state_dict or 'image_proj' in faceid_state_dict:
            proj_key = 'proj' if 'proj' in faceid_state_dict else 'image_proj'
            proj_weights = faceid_state_dict[proj_key]
            if isinstance(proj_weights, dict) and 'weight' in proj_weights:
                self.unet.encoder_hid_proj_faceid.load_state_dict(proj_weights)
                print(f"  Loaded projection weights from state_dict")

        # 3. Load LoRA weights
        lora_path = model_path / lora_weight_name
        if lora_path.exists():
            print(f"  Loading LoRA: {lora_path.name}")
            lora_state_dict = safetensors.torch.load_file(lora_path)
            # Apply LoRA to UNet (implementation depends on diffusers version)
            # For now, we'll store it for manual application
            self._faceid_lora_state_dict = lora_state_dict
            print(f"  LoRA weights loaded ({len(lora_state_dict)} keys)")
        else:
            print(f"  [WARN] LoRA not found: {lora_path}, skipping")
            self._faceid_lora_state_dict = None

        # 4. Initialize FaceIDAdapter (InsightFace wrapper)
        print(f"  Initializing FaceIDAdapter on {device}")
        self.faceid_adapter = FaceIDAdapter(device=device)

        # 5. Initialize CategoryDetector
        print(f"  Initializing CategoryDetector")
        self.category_detector = CategoryDetector(
            faceid_adapter=self.faceid_adapter,
            text_priority=True,
            image_fallback=True
        )

        self._faceid_loaded = True
        print(f"[OK] IP-Adapter FaceID loaded successfully!")

    def _prepare_faceid_embeddings(
        self,
        face_image,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance
    ):
        """
        Prepare FaceID embeddings for a single image.

        Args:
            face_image: PIL Image containing a face
            device: Target device
            num_images_per_prompt: Batch size
            do_classifier_free_guidance: Whether to use CFG

        Returns:
            torch.Tensor: FaceID embeddings ready for cross-attention [batch, 1, cross_attention_dim]
        """
        # Extract 512-dim face embedding using InsightFace
        try:
            face_embedding = self.faceid_adapter.extract_face_embedding(face_image)  # [512]
            print(f"[DEBUG] Extracted face embedding: shape = {face_embedding.shape}")
        except ValueError as e:
            print(f"[WARN] FaceID extraction failed: {e}")
            print(f"  Falling back to IP-Adapter Plus for this region")
            # Fallback to Plus if face detection fails
            return self.prepare_regional_ip_adapter_image_embeds(
                face_image, device, num_images_per_prompt, do_classifier_free_guidance
            )

        # Project to cross_attention_dim
        face_embedding = face_embedding.unsqueeze(0).to(device, dtype=self.unet.dtype)  # [1, 512]
        print(f"[DEBUG] Before projection: shape = {face_embedding.shape}")
        print(f"[DEBUG] Projection layer: {self.unet.encoder_hid_proj_faceid}")
        projected_embedding = self.unet.encoder_hid_proj_faceid(face_embedding)  # [1, cross_attention_dim]
        print(f"[DEBUG] After projection: shape = {projected_embedding.shape}")
        projected_embedding = projected_embedding.to(dtype=self.unet.dtype)  # Ensure correct dtype (float16)

        # Add token dimension (FaceID uses single token)
        projected_embedding = projected_embedding.unsqueeze(1)  # [1, 1, cross_attention_dim]
        print(f"[DEBUG] After unsqueeze: shape = {projected_embedding.shape}")

        # Repeat for batch
        if num_images_per_prompt > 1:
            projected_embedding = projected_embedding.repeat(num_images_per_prompt, 1, 1)
            print(f"[DEBUG] After repeat (num_images={num_images_per_prompt}): shape = {projected_embedding.shape}")

        # Handle CFG (classifier-free guidance)
        if do_classifier_free_guidance:
            # Negative embedding: zeros
            negative_embedding = torch.zeros_like(projected_embedding)
            projected_embedding = torch.cat([negative_embedding, projected_embedding], dim=0)
            print(f"[DEBUG] After CFG concat: shape = {projected_embedding.shape}")

        return projected_embedding

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds
    def prepare_regional_ip_adapter_image_embeds(
        self, ip_adapter_image, device, num_images_per_prompt, do_classifier_free_guidance,
        region_prompt: Optional[str] = None
    ):
        """
        Prepare image embeddings for regional IP-Adapter.
        Applies FULL projection pipeline for use in cross_attention.

        If FaceID is loaded and the region contains a person, uses FaceID instead of Plus.

        Args:
            ip_adapter_image: Reference image for this region
            device: Target device
            num_images_per_prompt: Batch size
            do_classifier_free_guidance: Whether to use CFG
            region_prompt: Text prompt for this region (used for category detection)

        Returns:
            Image embeddings ready for cross-attention
        """
        # Check if this is a person region and FaceID is available
        if self._faceid_loaded and region_prompt is not None and self.category_detector is not None:
            is_person, reason = self.category_detector.is_person_region(
                prompt=region_prompt,
                image=ip_adapter_image,
                verbose=False
            )

            if is_person:
                print(f"[RegionalIP] Person detected ({reason}), using FaceID")
                return self._prepare_faceid_embeddings(
                    ip_adapter_image, device, num_images_per_prompt, do_classifier_free_guidance
                )
            else:
                print(f"[RegionalIP] Non-person ({reason}), using Plus")

        # Default: use IP-Adapter Plus (original implementation below)
        if not isinstance(ip_adapter_image, list):
            ip_adapter_image = [ip_adapter_image]

        if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
            raise ValueError(
                f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
            )

        # Get projection layers
        image_proj_layer = self.unet.encoder_hid_proj.image_projection_layers[0]
        output_hidden_state = not isinstance(image_proj_layer, ImageProjection)

        # Step 1: Encode image (returns CLIP hidden states - 1664 dim)
        # Shape: [1, num_tokens, 1664]
        single_image_embeds, single_negative_image_embeds = self.encode_image(
            ip_adapter_image[0], device, 1, output_hidden_state
        )

        # Step 2: Apply CLIP's visual projection (1664 -> 1280) if hidden states are used
        if hasattr(self.image_encoder, 'visual_projection') and single_image_embeds.ndim == 3:
            batch_size, num_tokens, hidden_dim = single_image_embeds.shape
            single_image_embeds = single_image_embeds.reshape(-1, hidden_dim)
            single_image_embeds = self.image_encoder.visual_projection(single_image_embeds)
            single_image_embeds = single_image_embeds.reshape(batch_size, num_tokens, -1)

            batch_size_neg, num_tokens_neg, hidden_dim_neg = single_negative_image_embeds.shape
            single_negative_image_embeds = single_negative_image_embeds.reshape(-1, hidden_dim_neg)
            single_negative_image_embeds = self.image_encoder.visual_projection(single_negative_image_embeds)
            single_negative_image_embeds = single_negative_image_embeds.reshape(batch_size_neg, num_tokens_neg, -1)

        # Step 3: Apply projection layer
        single_image_embeds = image_proj_layer(single_image_embeds)
        single_negative_image_embeds = image_proj_layer(single_negative_image_embeds)

        # Ensure token dimension for attention (use 1 token for 2D embeddings)
        if single_image_embeds.ndim == 2:
            single_image_embeds = single_image_embeds.unsqueeze(1)
            single_negative_image_embeds = single_negative_image_embeds.unsqueeze(1)

        # Repeat for batch
        if num_images_per_prompt > 1:
            single_image_embeds = single_image_embeds.repeat(num_images_per_prompt, 1, 1)
            single_negative_image_embeds = single_negative_image_embeds.repeat(num_images_per_prompt, 1, 1)

        if do_classifier_free_guidance:
            single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

        single_image_embeds = single_image_embeds.to(device)

        return single_image_embeds

    def enable_multi_adapter_mode(self):
        """启用Multi-Adapter模式，并自动注册已加载的adapters"""
        if not hasattr(self, '_multi_adapter_enabled'):
            raise RuntimeError("Multi-Adapter not initialized. Call _init_multi_adapter() first.")

        # 自动注册已加载的IP-Adapter Plus
        if hasattr(self.unet, 'encoder_hid_proj') and self.unet.encoder_hid_proj is not None:
            print("[RegionalDiffusion] Registering IP-Adapter Plus to Multi-Adapter registry...")
            # 获取Plus adapter的image projection model
            plus_proj = self.unet.encoder_hid_proj

            # 注册Plus adapter（使用空的weights dict，因为Plus是通过attention processors工作的）
            self.multi_adapter_registry.register_adapter(
                name="plus",
                adapter_weights={},  # Plus adapter权重已经在attention processors中
                image_proj_model=plus_proj,
                scale=getattr(self, '_ip_adapter_scale', 0.7),  # 使用之前设置的scale
                adapter_type="plus"
            )

        # 自动注册已加载的IP-Adapter FaceID
        if hasattr(self, 'faceid_adapter') and self.faceid_adapter is not None:
            print("[RegionalDiffusion] Registering IP-Adapter FaceID to Multi-Adapter registry...")
            # 获取FaceID的projection model
            faceid_proj = self.unet.encoder_hid_proj_faceid if hasattr(self.unet, 'encoder_hid_proj_faceid') else None

            if faceid_proj is not None:
                # 注册FaceID adapter
                self.multi_adapter_registry.register_adapter(
                    name="faceid",
                    adapter_weights={'faceid_adapter': self.faceid_adapter},
                    image_proj_model=faceid_proj,
                    scale=0.8,  # FaceID默认scale
                    adapter_type="faceid"
                )

        self._multi_adapter_enabled = True
        print("[RegionalDiffusion] Multi-Adapter mode ENABLED")

        # 显示已注册的adapters
        registered = list(self.multi_adapter_registry.adapters.keys())
        if registered:
            print(f"[RegionalDiffusion] Registered adapters: {registered}")
        else:
            print("[RegionalDiffusion] WARNING: No adapters registered! Load adapters first.")

    def disable_multi_adapter_mode(self):
        """禁用Multi-Adapter模式（回退到原有实现）"""
        self._multi_adapter_enabled = False
        print("[RegionalDiffusion] Multi-Adapter mode DISABLED")

    def is_multi_adapter_enabled(self) -> bool:
        """检查Multi-Adapter模式是否启用"""
        return getattr(self, '_multi_adapter_enabled', False)

    def set_ip_adapter_scale(self, scale):
        """Override to save the scale value for Multi-Adapter mode"""
        # 保存scale值供Multi-Adapter使用
        self._ip_adapter_scale = scale if not isinstance(scale, list) else scale[0]

        # 调用父类方法设置实际的adapter scale
        super().set_ip_adapter_scale(scale)

    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            image_embeds = []
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                # Apply CLIP's visual projection if available and needed (1664 -> 1280)
                # Only apply if embeddings are 3D (hidden states mode)
                if hasattr(self.image_encoder, 'visual_projection') and single_image_embeds.ndim == 3:
                    batch_size, num_tokens, hidden_dim = single_image_embeds.shape
                    single_image_embeds = single_image_embeds.reshape(-1, hidden_dim)
                    single_image_embeds = self.image_encoder.visual_projection(single_image_embeds)
                    single_image_embeds = single_image_embeds.reshape(batch_size, num_tokens, -1)

                    batch_size_neg, num_tokens_neg, hidden_dim_neg = single_negative_image_embeds.shape
                    single_negative_image_embeds = single_negative_image_embeds.reshape(-1, hidden_dim_neg)
                    single_negative_image_embeds = self.image_encoder.visual_projection(single_negative_image_embeds)
                    single_negative_image_embeds = single_negative_image_embeds.reshape(batch_size_neg, num_tokens_neg, -1)

                single_image_embeds = torch.stack([single_image_embeds] * num_images_per_prompt, dim=0)
                single_negative_image_embeds = torch.stack(
                    [single_negative_image_embeds] * num_images_per_prompt, dim=0
                )

                if do_classifier_free_guidance:
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
                    single_image_embeds = single_image_embeds.to(device)

                image_embeds.append(single_image_embeds)
        else:
            repeat_dims = [1]
            image_embeds = []
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                    single_negative_image_embeds = single_negative_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_negative_image_embeds.shape[1:]))
                    )
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
                else:
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                image_embeds.append(single_image_embeds)

        return image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
                FusedAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.FloatTensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb
    def regional_info(self,prompts):
        ppl = prompts.split('BREAK')
        # print('ppl',ppl)
        targets =[p.split(",")[-1] for p in ppl[:]]
        # print('targets',targets)
        pt, ppt = [], []
        padd = 0
        for pp in targets:
            pp=pp.split(" ")
            pp=[p for p in pp if p != ""]
            tokensnum = len(pp)
            pt.append([padd, tokensnum // TOKENS + 1 + padd])
            ppt.append(tokensnum)
            padd = tokensnum // TOKENS + 1 + padd
        self.pt = pt
        self.ppt = ppt
        # print('pt:',self.pt)
        # print('ppt:',self.ppt)
    def torch_fix_seed(self, seed=42):
        # Python random
        random.seed(seed)
        # Numpy
        np.random.seed(seed)
        # Pytorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True
    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def denoising_end(self):
        return self._denoising_end

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        split_ratio: str,
        base_ratio: Optional[float] = None,
        base_prompt: Optional[str] = None,
        batch_size: Optional[int] = 1,
        prompt: Union[str, List[str]] = None,
        seed: Optional[int] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        regional_ip_adapter_images: Optional[List[PipelineImageInput]] = None,  # NEW: region-wise IP-Adapter
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],

        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise as determined by the discrete timesteps selected by the
                scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
                "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.FloatTensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """
        self.usebase = True if base_ratio is not None and base_prompt is not None else False
        self.base_ratio = base_ratio
        self.base_prompt = base_prompt
        self.split_ratio = split_ratio
        self.prompt = prompt if base_prompt is None else base_prompt+' BREAK '+prompt
        self.original_prompt = self.prompt
        self.h = height
        self.w = width
        self.pn = True
        self.eq = True
        self.isvanilla = True
        self.count = 0
        self.isxl = True
        self.batch_size = batch_size
        self.regional_info(self.prompt) 
        keyconverter(self,self.split_ratio,self.usebase)
        matrixdealer(self,self.split_ratio,self.base_ratio)
        if not hasattr(self, "_region_mask_feather"):
            self._region_mask_feather = 2
        if not hasattr(self, "_region_mask_feather_max"):
            self._region_mask_feather_max = 8
        if not hasattr(self, "_region_mask_feather_mode"):
            self._region_mask_feather_mode = "timestep"
        if not hasattr(self, "_region_masks_pyramid"):
            self._region_masks_pyramid = {}
        if seed is not None and seed > 0:
            self.torch_fix_seed(seed=seed)
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False

        # 2. Define call parameters
        if self.prompt is not None and isinstance(self.prompt, str):
            batch_size = 1
        elif self.prompt is not None and isinstance(self.prompt, list):
            batch_size = len(self.prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=self.original_prompt,
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
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

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

        # Handle IP-Adapter: global or region-wise
        regional_image_embeds_list = None
        if regional_ip_adapter_images is not None:
            # Check if using Multi-Adapter mode
            if self._multi_adapter_enabled:
                print("[MultiAdapter Mode] Preparing multi-adapter image embeds")

                regional_prompts_list = []
                if isinstance(prompt, str):
                    regional_prompts_list = [p.strip() for p in prompt.split("BREAK")]

                region_image_embeds_dict = self.prepare_multi_adapter_image_embeds(
                    regional_images=regional_ip_adapter_images,
                    regional_prompts=regional_prompts_list,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    do_classifier_free_guidance=self.do_classifier_free_guidance
                )

                num_regions = len(regional_ip_adapter_images)
                if len(region_image_embeds_dict) == 0:
                    raise ValueError("Multi-adapter returned no image embeddings for regions.")

                first_embed = next(iter(region_image_embeds_dict.values()))
                region_image_embeds_list = []
                for i in range(num_regions):
                    emb = region_image_embeds_dict.get(i)
                    if emb is None:
                        emb = torch.zeros_like(first_embed)
                    region_image_embeds_list.append(emb)

                self._regional_image_embeds = region_image_embeds_list

                region_scales = []
                if self.region_adapter_mapping is not None:
                    for i in range(num_regions):
                        adapter_name = self.region_adapter_mapping.get_adapter_for_region(i)
                        adapter_info = self.multi_adapter_registry.get_adapter(adapter_name) if adapter_name else None
                        region_scales.append(adapter_info['scale'] if adapter_info else 1.0)
                else:
                    region_scales = [1.0] * num_regions

                self._ip_scale = region_scales

                dummy_hidden_dim = None
                if hasattr(self.unet, 'encoder_hid_proj') and self.unet.encoder_hid_proj is not None:
                    proj_layer = self.unet.encoder_hid_proj.image_projection_layers[0]
                    if hasattr(proj_layer, 'proj_in') and hasattr(proj_layer.proj_in, 'in_features'):
                        dummy_hidden_dim = proj_layer.proj_in.in_features
                    elif hasattr(proj_layer, 'in_features'):
                        dummy_hidden_dim = proj_layer.in_features

                if dummy_hidden_dim is None:
                    dummy_hidden_dim = region_image_embeds_list[0].shape[-1]

                batch_size_total = region_image_embeds_list[0].shape[0]
                num_tokens = 4
                image_embeds = torch.zeros(
                    batch_size_total, 1, num_tokens, dummy_hidden_dim,
                    dtype=prompt_embeds.dtype, device=device
                )
                print(f"[DEBUG] Created dummy image_embeds: shape = {image_embeds.shape}")
            else:
                # Original single-adapter mode
                print("[SingleAdapter Mode] Using original region-wise IP-Adapter")

                # NEW: Region-wise IP-Adapter
                # Prepare FULLY PROJECTED embeddings for each region (for cross_attention)
                # Split prompt to get individual regional prompts for category detection
                regional_prompts = prompt.split('BREAK') if isinstance(prompt, str) else []

                regional_image_embeds_list = []
                for i, regional_img in enumerate(regional_ip_adapter_images):
                    # Get corresponding regional prompt (or None if not available)
                    current_region_prompt = regional_prompts[i].strip() if i < len(regional_prompts) else None

                    regional_embeds = self.prepare_regional_ip_adapter_image_embeds(
                        regional_img,
                        device,
                        batch_size * num_images_per_prompt,
                        self.do_classifier_free_guidance,
                        region_prompt=current_region_prompt,  # NEW: pass regional prompt for category detection
                    )
                    regional_image_embeds_list.append(regional_embeds)

                # Also prepare RAW embeddings for first region as dummy (for UNet check)
                # This will be processed by UNet but ignored in cross_attention
                dummy_raw_embeds = self.prepare_ip_adapter_image_embeds(
                    regional_ip_adapter_images[0],
                    None,
                    device,
                    batch_size * num_images_per_prompt,
                    self.do_classifier_free_guidance,
                )

                if isinstance(dummy_raw_embeds, list):
                    dummy_raw_embeds = dummy_raw_embeds[0]

                # Get IP-Adapter scale from attention processors
                ip_scale = 1.0  # default
                for attn_processor in self.unet.attn_processors.values():
                    if hasattr(attn_processor, 'scale') and len(attn_processor.scale) > 0:
                        # Get the first scale value
                        ip_scale = attn_processor.scale[0] if isinstance(attn_processor.scale[0], (int, float)) else 1.0
                        break

                # Store as instance variables for use in attention layers
                self._regional_image_embeds = regional_image_embeds_list
                self._ip_scale = ip_scale
                image_embeds = dummy_raw_embeds  # Use dummy for UNet, will be overridden in cross_attention
        elif ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            # Original: Global IP-Adapter
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )
        else:
            image_embeds = None

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 8.1 Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)
        if not hasattr(self, "_ip_timestep_start"):
            self._ip_timestep_start = 0.0
        if not hasattr(self, "_ip_timestep_end"):
            self._ip_timestep_end = 0.6
        if not hasattr(self, "_ip_timestep_mid"):
            self._ip_timestep_mid = 0.3
        if not hasattr(self, "_ip_timestep_early_scale"):
            self._ip_timestep_early_scale = 0.0
        if not hasattr(self, "_ip_timestep_mid_scale"):
            self._ip_timestep_mid_scale = 1.0
        if not hasattr(self, "_ip_timestep_late_scale"):
            self._ip_timestep_late_scale = 0.2
        if not hasattr(self, "_ip_timestep_mode"):
            self._ip_timestep_mode = "piecewise"
        if not hasattr(self, "_ip_disable_up"):
            self._ip_disable_up = True
        if not hasattr(self, "_ip_norm_cap_ratio"):
            self._ip_norm_cap_ratio = 0.6
        if not hasattr(self, "_ip_norm_cap_eps"):
            self._ip_norm_cap_eps = 1e-6
        if not hasattr(self, "_ip_entropy_gate"):
            self._ip_entropy_gate = 0.4
        if not hasattr(self, "_ip_entropy_gate_eps"):
            self._ip_entropy_gate_eps = 1e-6
        if not hasattr(self, "_ip_use_value_only"):
            self._ip_use_value_only = True
        if not hasattr(self, "_ip_router_enabled"):
            self._ip_router_enabled = True
        if not hasattr(self, "_ip_router_default"):
            self._ip_router_default = 0.0
        if not hasattr(self, "_ip_router_scale_on_hit"):
            self._ip_router_scale_on_hit = 1.0
        if not hasattr(self, "_ip_router_keywords"):
            self._ip_router_keywords = [
                "person",
                "man",
                "woman",
                "girl",
                "boy",
                "face",
                "portrait",
                "people",
                "human",
                "character",
            ]
        if not hasattr(self, "_ip_router_prompt"):
            self._ip_router_prompt = None
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Provide timestep info for IP-Adapter gating in attention hooks
                self._ip_denoising_step = i
                self._ip_denoising_steps = num_inference_steps

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                # Add image_embeds (dummy for regional, actual for global)
                if image_embeds is not None:
                    print(f"[DEBUG] image_embeds type: {type(image_embeds)}, shape: {image_embeds.shape if hasattr(image_embeds, 'shape') else 'N/A'}")
                    if isinstance(image_embeds, list):
                        added_cond_kwargs["image_embeds"] = image_embeds
                    else:
                        # Wrap in list to match expected format
                        added_cond_kwargs["image_embeds"] = [image_embeds]
                # Note: For regional IP, dummy goes through UNet but regional_image_embeds
                # stored in self._regional_image_embeds will override in cross_attention hooks
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            # MODIFIED: Force full VAE float32 conversion to avoid dtype mismatches with FaceID
            needs_upcasting = self.vae.dtype == torch.float16

            if needs_upcasting:
                # Force entire VAE to float32 (no selective downgrading)
                self.vae.to(dtype=torch.float32)
                latents = latents.to(dtype=torch.float32)
            elif latents.dtype != self.vae.dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    self.vae = self.vae.to(latents.dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Clean up regional IP-Adapter variables
        if hasattr(self, '_regional_image_embeds'):
            self._regional_image_embeds = None
        if hasattr(self, '_ip_scale'):
            self._ip_scale = None
        if hasattr(self, "_ip_denoising_step"):
            self._ip_denoising_step = None
        if hasattr(self, "_ip_denoising_steps"):
            self._ip_denoising_steps = None

        # Clean up Multi-Adapter hooks and variables
        if hasattr(self, 'spatial_aware_hooks') and len(self.spatial_aware_hooks) > 0:
            remove_spatial_aware_hooks(self.spatial_aware_hooks)
            self.spatial_aware_hooks = []

        # Clean up Multi-Adapter temporary variables
        self.spatial_aware_adapter = None
        if hasattr(self, '_region_masks'):
            self._region_masks = None
        if hasattr(self, "_region_masks_pyramid"):
            self._region_masks_pyramid = {}
        if hasattr(self, '_region_image_embeds_dict'):
            self._region_image_embeds_dict = None

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)

# pipe = RegionalDiffusionXLPipeline.from_single_file("../models/RPG_models/albedobaseXL_v20.safetensors",torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
# # pipe = RegionalDiffusionXLPipeline.from_pretrained("../models/stable-diffusion-xl-base-1.0", use_safetensors=True, torch_dtype=torch.float32)
# pipe.to("cuda")
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
# pipe.enable_xformers_memory_efficient_attention()
# prompt= 'blue sky BREAK black hair twintail BREAK aquarium BREAK messy desk BREAK orange dress and sofa'
# split_ratio= '1,2,1,1;2,4,6'
# negative_prompt = ''
# images = pipe(
#     prompt=prompt,
#     split_ratio=split_ratio, # The ratio of the regional prompt, the number of prompts is the same as the number of regions, and the number of prompts is the same as the number of regions
#     batch_size = 1, #batch size
#     base_ratio = 0, # The ratio of the base prompt    
#     base_prompt= None,       
#     num_inference_steps=20, # sampling step
#     height = 1024, 
#     negative_prompt=negative_prompt, # negative prompt
#     width = 1024, 
#     seed = 1234,# random seed
#     guidance_scale = 7.0
# ).images[0]
# images.save("albedobaseXL_1.png")

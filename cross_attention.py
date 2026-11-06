import math
from pprint import pprint
import torch
import torch.nn.functional as F

# Make xformers optional
try:
    import xformers
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    print("Warning: xformers not installed. Using standard attention (slower but works).")

TOKENSCON = 77
TOKENS = 75

DEFAULT_IP_LAYER_SCALES = {
    "down_blocks.0": 0.2,
    "down_blocks.1": 0.2,
    "down_blocks.2": 0.6,
    "mid_block": 0.6,
    "up_blocks.2": 0.4,
    "up_blocks.3": 0.4,
}


def pool_ip_hidden_states(ip_hidden_states, target_len: int):
    if ip_hidden_states is None:
        return None
    if ip_hidden_states.shape[1] == target_len:
        return ip_hidden_states

    # Pool along the token dimension to match text length.
    # [B, T, D] -> [B, D, T] -> pool -> [B, T_target, D]
    pooled = ip_hidden_states.transpose(1, 2)
    pooled = F.adaptive_avg_pool1d(pooled, target_len)
    return pooled.transpose(1, 2)


def resolve_layer_ip_scale(pipeline, layer_name: str) -> float:
    if not layer_name:
        return 1.0

    if getattr(pipeline, "_ip_disable_up", False) and "up_blocks" in layer_name:
        return 0.0

    layer_scales = getattr(pipeline, "_ip_layer_scales", None)
    if callable(layer_scales):
        return float(layer_scales(layer_name))
    if isinstance(layer_scales, dict):
        for key, value in layer_scales.items():
            if key in layer_name:
                return float(value)

    for key, value in DEFAULT_IP_LAYER_SCALES.items():
        if key in layer_name:
            return float(value)

    return 0.4


def resolve_router_ip_scale(pipeline, prompt: str = None) -> float:
    if not getattr(pipeline, "_ip_router_enabled", False):
        return 1.0

    forced = getattr(pipeline, "_ip_router_force", None)
    if forced is not None:
        try:
            return float(forced)
        except (TypeError, ValueError):
            return 1.0

    prompt_text = getattr(pipeline, "_ip_router_prompt", None) or prompt
    if isinstance(prompt_text, (list, tuple)):
        prompt_text = " ".join([str(p) for p in prompt_text if p is not None])
    if not prompt_text:
        return float(getattr(pipeline, "_ip_router_default", 1.0))

    keywords = getattr(pipeline, "_ip_router_keywords", None)
    if not keywords:
        return float(getattr(pipeline, "_ip_router_default", 1.0))

    prompt_lower = str(prompt_text).lower()
    for kw in keywords:
        if kw and str(kw).lower() in prompt_lower:
            return float(getattr(pipeline, "_ip_router_scale_on_hit", 1.0))

    return float(getattr(pipeline, "_ip_router_default", 0.0))


def apply_ip_norm_cap(text_out, ip_out, cap_ratio: float, eps: float):
    if cap_ratio is None or cap_ratio <= 0.0:
        return ip_out

    text_norm = torch.norm(text_out, dim=(1, 2), keepdim=True)
    ip_norm = torch.norm(ip_out, dim=(1, 2), keepdim=True)
    scale = cap_ratio * text_norm / (ip_norm + eps)
    scale = torch.clamp(scale, max=1.0)
    return ip_out * scale


def apply_ip_entropy_gate(attn_weights, ip_out, gate_strength: float, eps: float):
    if gate_strength is None or gate_strength <= 0.0:
        return ip_out

    # attn_weights: [batch, heads, query_len, key_len]
    # Compute normalized entropy per batch averaged across heads/queries.
    log_p = torch.log(attn_weights + eps)
    entropy = -torch.sum(attn_weights * log_p, dim=-1)  # [batch, heads, query_len]
    ent = entropy.mean(dim=(1, 2), keepdim=True)  # [batch, 1, 1]

    key_len = attn_weights.shape[-1]
    max_ent = math.log(max(key_len, 2))
    norm_ent = ent / max_ent
    gate = torch.clamp(1.0 - gate_strength * norm_ent, 0.0, 1.0)
    return ip_out * gate


def resolve_timestep_ip_scale(pipeline) -> float:
    step = getattr(pipeline, "_ip_denoising_step", None)
    total_steps = getattr(pipeline, "_ip_denoising_steps", None)
    if step is None or total_steps is None or total_steps < 2:
        return 1.0

    start = float(getattr(pipeline, "_ip_timestep_start", 0.0))
    end = float(getattr(pipeline, "_ip_timestep_end", 0.6))
    progress = float(step) / float(total_steps - 1)

    mode = str(getattr(pipeline, "_ip_timestep_mode", "cosine")).lower()
    if mode == "piecewise":
        mid = float(getattr(pipeline, "_ip_timestep_mid", 0.3))
        early_scale = float(getattr(pipeline, "_ip_timestep_early_scale", 0.0))
        mid_scale = float(getattr(pipeline, "_ip_timestep_mid_scale", 1.0))
        late_scale = float(getattr(pipeline, "_ip_timestep_late_scale", 0.2))

        if end <= start:
            return late_scale
        mid = max(min(mid, end), start)

        if progress <= start:
            return early_scale
        if progress <= mid:
            denom = max(mid - start, 1e-6)
            t = (progress - start) / denom
            return early_scale + t * (mid_scale - early_scale)
        if progress <= end:
            denom = max(end - mid, 1e-6)
            t = (progress - mid) / denom
            return mid_scale + t * (late_scale - mid_scale)
        return late_scale

    if end <= start:
        return 1.0
    if progress <= start:
        return 1.0
    if progress >= end:
        return 0.0

    t = (progress - start) / (end - start)
    if mode == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * t))
    return 1.0 - t


def scale_ip_scale(ip_scale, scale: float):
    if isinstance(ip_scale, (list, tuple)):
        return [s * scale for s in ip_scale]
    return ip_scale * scale


def build_region_masks_from_split_ratio(split_ratio, latent_h, latent_w, feather_pixels, device):
    masks = []
    if split_ratio is None:
        return masks

    y = torch.arange(latent_h, device=device).view(latent_h, 1).expand(latent_h, latent_w)
    x = torch.arange(latent_w, device=device).view(1, latent_w).expand(latent_h, latent_w)
    feather = max(int(feather_pixels), 0)
    for drow in split_ratio:
        row_start = int(latent_h * drow.start)
        row_end = int(latent_h * drow.end)
        for dcell in drow.cols:
            col_start = int(latent_w * dcell.start)
            col_end = int(latent_w * dcell.end)

            inside = (
                (y >= row_start)
                & (y < row_end)
                & (x >= col_start)
                & (x < col_end)
            )
            if feather == 0:
                mask = inside.float()
            else:
                dist_left = x - col_start
                dist_right = (col_end - 1) - x
                dist_top = y - row_start
                dist_bottom = (row_end - 1) - y
                dist = torch.minimum(
                    torch.minimum(dist_left, dist_right),
                    torch.minimum(dist_top, dist_bottom),
                )
                weight = torch.clamp(dist.float() / float(feather), 0.0, 1.0)
                mask = weight * inside.float()

            masks.append(mask)
    return masks


def resolve_region_feather(pipeline) -> int:
    base = int(getattr(pipeline, "_region_mask_feather", 2))
    mode = str(getattr(pipeline, "_region_mask_feather_mode", "static")).lower()
    if mode in ("static", "fixed"):
        return base

    max_feather = int(getattr(pipeline, "_region_mask_feather_max", base))
    if max_feather <= base:
        return base

    step = getattr(pipeline, "_ip_denoising_step", None)
    total_steps = getattr(pipeline, "_ip_denoising_steps", None)
    if step is None or total_steps is None or total_steps < 2:
        return base

    progress = float(step) / float(total_steps - 1)
    if mode == "cosine":
        weight = 0.5 * (1.0 - math.cos(math.pi * progress))
    else:
        weight = progress

    return int(round(base + (max_feather - base) * weight))


def get_region_masks_for_latent(pipeline, latent_h, latent_w, device):
    split_ratio = getattr(pipeline, "split_ratio", None)
    if split_ratio is None:
        return None

    feather = resolve_region_feather(pipeline)
    pyramid = getattr(pipeline, "_region_masks_pyramid", None)
    if pyramid is None:
        pyramid = {}
        pipeline._region_masks_pyramid = pyramid

    key = (device.type, latent_h, latent_w, feather)
    if key not in pyramid:
        pyramid[key] = build_region_masks_from_split_ratio(
            split_ratio, latent_h, latent_w, feather, device
        )
    return pyramid.get(key)



def _memory_efficient_attention_xformers(module, query, key, value):
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    if XFORMERS_AVAILABLE:
        # Use xformers if available (faster)
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)
    else:
        # Fallback to standard attention
        # Compute attention scores
        scale = 1 / math.sqrt(query.shape[-1])
        attention_scores = torch.bmm(query, key.transpose(-1, -2)) * scale
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        hidden_states = torch.bmm(attention_probs, value)

    hidden_states = module.batch_to_head_dim(hidden_states)
    return hidden_states


def apply_ip_adapter_attention(
    module,
    query,
    ip_hidden_states,
    scale=1.0,
    text_key=None,
    use_ip_value_only=False,
):
    """
    Apply IP-Adapter attention using image embeddings.
    This adds image-based attention to the existing text-based attention output.

    IMPORTANT: This follows the standard IPAdapterAttnProcessor2_0 implementation.

    Args:
        module: The attention module
        query: Query tensor [batch, seq_len, dim] (BEFORE head_to_batch_dim)
        ip_hidden_states: Image embeddings [batch, num_tokens, hidden_dim]
        scale: IP-Adapter scale factor

    Returns:
        IP attention output [batch, seq_len, dim] to be added to text attention output
    """
    # Check if module has IP-Adapter processors
    if not hasattr(module, 'processor') or not hasattr(module.processor, 'to_k_ip'):
        return None, None

    processor = module.processor
    # Get batch_size from query, NOT from ip_hidden_states
    # query shape: [batch, seq_len, dim]
    batch_size = query.shape[0]

    # Generate IP key and value using dedicated IP projections
    # processor.to_k_ip and to_v_ip are ModuleList, use first one
    to_k_ip = processor.to_k_ip[0] if isinstance(processor.to_k_ip, torch.nn.ModuleList) else processor.to_k_ip
    to_v_ip = processor.to_v_ip[0] if isinstance(processor.to_v_ip, torch.nn.ModuleList) else processor.to_v_ip

    if use_ip_value_only and text_key is not None:
        pooled = pool_ip_hidden_states(ip_hidden_states, text_key.shape[1])
        ip_key = text_key
        ip_value = to_v_ip(pooled)
    else:
        ip_key = to_k_ip(ip_hidden_states)
        ip_value = to_v_ip(ip_hidden_states)

    # Get dimensions
    inner_dim = ip_key.shape[-1]
    head_dim = inner_dim // module.heads

    # Reshape key and value for multi-head attention
    # Standard implementation: view + transpose
    ip_key = ip_key.view(batch_size, -1, module.heads, head_dim).transpose(1, 2)
    ip_value = ip_value.view(batch_size, -1, module.heads, head_dim).transpose(1, 2)

    # Reshape query for multi-head attention
    # query input: [batch, seq_len, dim]
    query_for_ip = query.view(batch_size, -1, module.heads, head_dim).transpose(1, 2)

    # Apply attention with explicit probs (needed for entropy gating)
    scale_factor = 1 / math.sqrt(head_dim)
    attn_scores = torch.matmul(query_for_ip, ip_key.transpose(-1, -2)) * scale_factor
    attn_probs = torch.softmax(attn_scores, dim=-1)
    current_ip_hidden_states = torch.matmul(attn_probs, ip_value)

    # Reshape output following standard implementation
    # transpose(1, 2): [batch, seq_len, heads, head_dim]
    # reshape: [batch, seq_len, heads * head_dim] = [batch, seq_len, dim]
    current_ip_hidden_states = current_ip_hidden_states.transpose(1, 2).reshape(
        batch_size, -1, module.heads * head_dim
    )
    current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)

    # Apply scale
    current_ip_hidden_states = current_ip_hidden_states * scale

    return current_ip_hidden_states, attn_probs


def apply_faceid_attention(module, query, faceid_hidden_states, scale=1.0):
    """
    Apply IP-Adapter FaceID attention using face embeddings.
    Similar to apply_ip_adapter_attention but handles FaceID embeddings.

    NOTE: Currently identical to apply_ip_adapter_attention since both FaceID and Plus
    embeddings are projected to the same cross_attention_dim in RegionalDiffusion_xl.py.
    Kept separate for clarity and future customization if needed.

    Args:
        module: The attention module
        query: Query tensor [batch, seq_len, dim]
        faceid_hidden_states: Face embeddings [batch, num_tokens, hidden_dim]
        scale: FaceID scale factor

    Returns:
        FaceID attention output [batch, seq_len, dim]
    """
    # Check if module has IP-Adapter processors (FaceID uses same processors)
    if not hasattr(module, 'processor') or not hasattr(module.processor, 'to_k_ip'):
        return None

    processor = module.processor
    batch_size = query.shape[0]

    # Use same IP projections (to_k_ip, to_v_ip) for FaceID
    # The difference is in the input embeddings (512-dim face vs 1280-dim CLIP)
    to_k_ip = processor.to_k_ip[0] if isinstance(processor.to_k_ip, torch.nn.ModuleList) else processor.to_k_ip
    to_v_ip = processor.to_v_ip[0] if isinstance(processor.to_v_ip, torch.nn.ModuleList) else processor.to_v_ip

    ip_key = to_k_ip(faceid_hidden_states)
    ip_value = to_v_ip(faceid_hidden_states)

    # Get dimensions
    inner_dim = ip_key.shape[-1]
    head_dim = inner_dim // module.heads

    # Reshape key and value for multi-head attention
    ip_key = ip_key.view(batch_size, -1, module.heads, head_dim).transpose(1, 2)
    ip_value = ip_value.view(batch_size, -1, module.heads, head_dim).transpose(1, 2)

    # Reshape query for multi-head attention
    query_for_ip = query.view(batch_size, -1, module.heads, head_dim).transpose(1, 2)

    # Apply scaled dot product attention
    current_ip_hidden_states = F.scaled_dot_product_attention(
        query_for_ip, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
    )

    # Reshape output
    current_ip_hidden_states = current_ip_hidden_states.transpose(1, 2).reshape(
        batch_size, -1, module.heads * head_dim
    )
    current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)

    # Apply scale
    current_ip_hidden_states = current_ip_hidden_states * scale

    return current_ip_hidden_states


def semantic_gate_map(module, query_batched, key_batched, mode="soft"):
    """Where this region's own text tokens are attending, at this layer.

    RPG already gates the adapter by the planner's rectangle.  A rectangle
    contains background as well as subject, so the reference's texture lands on
    the wall behind the person as readily as on the person.  Conceptrol's
    observation is that the text cross-attention of the region's own tokens
    already draws the subject's outline, and it is available here for free:
    the same query and key that produce the text branch produce it.

    Returns [batch, HW, 1] in [0,1], peaked on the subject.
    """
    d = query_batched.shape[-1]
    scores = torch.matmul(query_batched, key_batched.transpose(-1, -2)) / math.sqrt(d)
    probs = torch.softmax(scores.float(), dim=-1)          # [B*h, HW, T]
    m = probs[:, :, 1:].mean(dim=-1)                        # drop BOS
    bh, hw = m.shape
    heads = max(1, int(getattr(module, "heads", 1)))
    m = m.view(bh // heads, heads, hw).mean(dim=1)          # average the heads
    m = m - m.amin(dim=-1, keepdim=True)
    m = m / (m.amax(dim=-1, keepdim=True) + 1e-6)
    if mode == "otsu":
        # Conceptrol binarises; a per-sample mean threshold is the cheap
        # equivalent and needs no histogram
        m = (m > m.mean(dim=-1, keepdim=True)).to(m.dtype)
    return m.unsqueeze(-1)


def main_forward_diffusers(module,hidden_states,encoder_hidden_states,divide,userpp = False,tokens=[],width = 64,height = 64,step = 0, isxl = False, inhr = None, regional_ip_embeds=None, ip_scale=1.0):
    context = encoder_hidden_states
    import os as _os
    if _os.environ.get("SEM_TRACE"):
        global _CTX_SHOWN
        try:
            _CTX_SHOWN
        except NameError:
            _CTX_SHOWN = 0
        if _CTX_SHOWN < 3:
            _CTX_SHOWN += 1
            print(f"[ctx] context={tuple(context.shape)} mod77={context.shape[1]%77} "
                  f"regional_ip_embeds={'None' if regional_ip_embeds is None else tuple(regional_ip_embeds.shape)} "
                  f"ip_scale={ip_scale}", flush=True)
    query = module.to_q(hidden_states)
    key = module.to_k(context)
    value = module.to_v(context)

    # Convert to batch*heads format for text attention
    query_batched = module.head_to_batch_dim(query)
    key_batched = module.head_to_batch_dim(key)
    value_batched = module.head_to_batch_dim(value)

    # Compute text-based attention
    # Note: _memory_efficient_attention_xformers already calls batch_to_head_dim internally
    text_hidden = _memory_efficient_attention_xformers(module, query_batched, key_batched, value_batched)
    text_hidden = text_hidden.to(query_batched.dtype)

    # Apply IP-Adapter on a separate branch if embeddings provided
    ip_hidden = None
    ip_attn_probs = None
    if regional_ip_embeds is not None and ip_scale != 0.0:
        use_value_only = bool(getattr(module, "_ip_use_value_only", False))
        ip_hidden, ip_attn_probs = apply_ip_adapter_attention(
            module,
            query,
            regional_ip_embeds,
            scale=ip_scale,
            text_key=key if use_value_only else None,
            use_ip_value_only=use_value_only,
        )

    text_out = module.to_out[0](text_hidden)
    text_out = module.to_out[1](text_out)

    if ip_hidden is not None:
        ip_out = module.to_out[0](ip_hidden.to(text_out.dtype))
        ip_out = module.to_out[1](ip_out)
        cap_ratio = float(getattr(module, "_ip_norm_cap_ratio", 0.0))
        cap_eps = float(getattr(module, "_ip_norm_cap_eps", 1e-6))
        ip_out = apply_ip_norm_cap(text_out, ip_out, cap_ratio, cap_eps)
        gate_strength = float(getattr(module, "_ip_entropy_gate", 0.0))
        gate_eps = float(getattr(module, "_ip_entropy_gate_eps", 1e-6))
        if ip_attn_probs is not None:
            ip_out = apply_ip_entropy_gate(ip_attn_probs, ip_out, gate_strength, gate_eps)
        sem = float(getattr(module, "_ip_semantic_gate", 0.0))
        import os as _os
        if _os.environ.get("SEM_TRACE"):
            global _SEM_HITS
            try:
                _SEM_HITS += 1
            except NameError:
                _SEM_HITS = 1
            if _SEM_HITS <= 3:
                print(f"[sem] ip branch reached, gate={sem}", flush=True)
        if sem > 0.0:
            mode = str(getattr(module, "_ip_semantic_mode", "soft"))
            m = semantic_gate_map(module, query_batched, key_batched, mode)
            m = m.to(ip_out.dtype)
            # sem interpolates between the rectangle alone (0) and the
            # rectangle intersected with the semantic map (1)
            ip_out = ip_out * ((1.0 - sem) + sem * m)
        return text_out + ip_out

    return text_out
    
    
    
    
    
def hook_forwards(self, root_module: torch.nn.Module):
    for name, module in root_module.named_modules():
        if "attn2" in name and module.__class__.__name__ == "Attention":
            # print(f"Attaching hook to {name}")
            module._rd_attn_name = name
            module.forward = hook_forward(self, module)           


def hook_forward(self, module):
    def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        x= hidden_states
        context= encoder_hidden_states

        # Handle IP-Adapter: context might be a tuple (text_embeds, image_embeds)
        regional_image_embeds = None
        if isinstance(context, tuple):
            # For global IP-Adapter: use only text embeddings for regional control
            # IP-Adapter's image embeddings will be handled globally
            context = context[0]

        # Check for regional IP-Adapter embeddings
        # These are stored as _regional_image_embeds in RegionalDiffusion_xl.py
        if hasattr(self, '_regional_image_embeds') and self._regional_image_embeds is not None:
            regional_image_embeds = self._regional_image_embeds

        # Get IP-Adapter scale if available
        ip_scale = getattr(self, '_ip_scale', 1.0)
        layer_name = getattr(module, "_rd_attn_name", "")
        layer_scale = resolve_layer_ip_scale(self, layer_name)
        timestep_scale = resolve_timestep_ip_scale(self)
        router_scale = resolve_router_ip_scale(self, getattr(self, "_ip_router_prompt", None))
        ip_scale = scale_ip_scale(ip_scale, layer_scale * timestep_scale * router_scale)
        module._ip_norm_cap_ratio = getattr(self, "_ip_norm_cap_ratio", 0.0)
        module._ip_norm_cap_eps = getattr(self, "_ip_norm_cap_eps", 1e-6)
        module._ip_entropy_gate = getattr(self, "_ip_entropy_gate", 0.0)
        module._ip_entropy_gate_eps = getattr(self, "_ip_entropy_gate_eps", 1e-6)
        module._ip_use_value_only = getattr(self, "_ip_use_value_only", False)
        module._ip_semantic_gate = getattr(self, "_ip_semantic_gate", 0.0)
        module._ip_semantic_mode = getattr(self, "_ip_semantic_mode", "soft")

        # print("input : ", hidden_states.size())
        # print("tokens : ", context.size())
        # print("module : ", getattr(module, self.name,None))
        height =self.h
        width =self.w
        x_t = x.size()[1]
        scale = round(math.sqrt(height * width / x_t))
        latent_h = round(height / scale)
        latent_w = round(width / scale)
        ha, wa = x_t % latent_h, x_t % latent_w
        if ha == 0:
            latent_w = int(x_t / latent_h)
        elif wa == 0:
            latent_h = int(x_t / latent_w)

        contexts = context.clone()

        def matsepcalc(x,contexts,pn,divide):
            h_states = []
            x_t = x.size()[1]
            (latent_h,latent_w) = split_dims(x_t, height, width, self)
            

            latent_out = latent_w
            latent_in = latent_h
            region_masks = get_region_masks_for_latent(self, latent_h, latent_w, x.device)
            region_mask_idx = 0
            # Index into _regional_image_embeds / per-region ip_scale.  This must
            # NOT be `i`: `i` walks the prompt-chunk list, which carries the base
            # prompt as element 0 when usebase is set, while the regional embeds
            # list has one entry per region and no base entry.  Using `i` shifted
            # every region by one, so region 0 was denoised with region 1's
            # reference and the last region received None.
            region_ip_idx = 0

            tll = self.pt
            
            i = 0
            outb = None
            if self.usebase:
                context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
                cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                if cnet_ext > 0:
                    context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)

                i = i + 1

                # For base, don't use regional IP (or use None/global)
                out = main_forward_diffusers(module, x, context, divide,userpp =True,isxl = self.isxl, regional_ip_embeds=None, ip_scale=ip_scale)

                # if self.usebase:
                outb = out.clone()
                outb = outb.reshape(outb.size()[0], latent_h, latent_w, outb.size()[2]) 

            sumout = 0

            for drow in self.split_ratio:
                v_states = []
                sumin = 0
                for dcell in drow.cols:
                    # Grabs a set of tokens depending on number of unrelated breaks.
                    context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
                    # Controlnet sends extra conds at the end of context, apply it to all regions.
                    cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                    if cnet_ext > 0:
                        context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)

                    # NEW: Get regional IP-Adapter image embeddings for this region
                    region_ip_emb = None
                    region_ip_scale = ip_scale
                    if isinstance(ip_scale, (list, tuple)):
                        if len(ip_scale) > 0:
                            region_ip_scale = (ip_scale[region_ip_idx]
                                               if region_ip_idx < len(ip_scale)
                                               else ip_scale[-1])
                        else:
                            region_ip_scale = 1.0

                    # Use self._regional_image_embeds directly to get the potentially chunked version
                    current_regional_embeds = getattr(self, '_regional_image_embeds', None)
                    if current_regional_embeds is not None and region_ip_idx < len(current_regional_embeds):
                        # Get the image embeddings for this region
                        # regional_image_embeds is a list of embeddings, one per region
                        region_ip_emb = current_regional_embeds[region_ip_idx]

                    i = i + 1 + dcell.breaks
                    region_ip_idx += 1
                    # if i >= contexts.size()[1]:
                    #     indlast = True

                    out = main_forward_diffusers(module, x, context, divide,userpp = self.pn, isxl = self.isxl, regional_ip_embeds=region_ip_emb, ip_scale=region_ip_scale)
                    
                    out = out.reshape(out.size()[0], latent_h, latent_w, out.size()[2]) # convert to main shape.
                    # if indlast:
                    addout = 0
                    addin = 0
                    sumin = sumin + int(latent_in*dcell.end) - int(latent_in*dcell.start)
                    if dcell.end >= 0.999:
                        addin = sumin - latent_in
                        sumout = sumout + int(latent_out*drow.end) - int(latent_out*drow.start)
                        if drow.end >= 0.999:
                            addout = sumout - latent_out
                    row_start = int(latent_h * drow.start) + addout
                    row_end = int(latent_h * drow.end)
                    col_start = int(latent_w * dcell.start) + addin
                    col_end = int(latent_w * dcell.end)
                    out = out[:, row_start:row_end, col_start:col_end, :]
                    if self.usebase : 
                        # outb_t = outb[:,:,int(latent_w*drow.start):int(latent_w*drow.end),:].clone()
                        outb_t = outb[:, row_start:row_end, col_start:col_end, :].clone()
                        if region_masks is not None and region_mask_idx < len(region_masks):
                            mask = region_masks[region_mask_idx][row_start:row_end, col_start:col_end]
                            mask = mask.to(out.dtype).unsqueeze(0).unsqueeze(-1)
                            out = out * mask + outb_t * (1 - mask)
                        out = out * (1 - dcell.base) + outb_t * dcell.base
                    if region_masks is not None:
                        region_mask_idx += 1
            
                    v_states.append(out)

                            
                output_x = torch.cat(v_states,dim = 2) # First concat the cells to rows.

                h_states.append(output_x)
            output_x = torch.cat(h_states,dim = 1) # Second, concat rows to layer.
            output_x = output_x.reshape(x.size()[0],x.size()[1],x.size()[2]) # Restore to 3d source.  
            return output_x
        if x.size()[0] == 1 * self.batch_size:
            output_x = matsepcalc(x, contexts, self.pn, 1)
        else:
            # Chunk CFG (positive and negative conditions)
            if self.isvanilla: # SBM Ddim reverses cond/uncond.
                nx, px = x.chunk(2)
                conn,conp = contexts.chunk(2)
            else:
                px, nx = x.chunk(2)
                conp,conn = contexts.chunk(2)

            # Also chunk regional IP embeddings if present
            if regional_image_embeds is not None:
                # regional_image_embeds is a list of embeddings, each with shape [2, tokens, dim] (CFG)
                # We need to chunk each one
                regional_image_embeds_positive = []
                regional_image_embeds_negative = []
                for emb in regional_image_embeds:
                    if emb.shape[0] == 2:  # Has CFG batch dimension
                        neg_emb, pos_emb = emb.chunk(2, dim=0)
                        regional_image_embeds_positive.append(pos_emb)
                        regional_image_embeds_negative.append(neg_emb)
                    else:
                        # Single batch, use as is for both
                        regional_image_embeds_positive.append(emb)
                        regional_image_embeds_negative.append(emb)

                # Store chunked embeddings
                original_embeds = self._regional_image_embeds

                # Process positive
                self._regional_image_embeds = regional_image_embeds_positive
                opx = matsepcalc(px, conp, True, 2)

                # Process negative
                self._regional_image_embeds = regional_image_embeds_negative
                onx = matsepcalc(nx, conn, False, 2)

                # Restore original
                self._regional_image_embeds = original_embeds
            else:
                opx = matsepcalc(px, conp, True, 2)
                onx = matsepcalc(nx, conn, False, 2)

            if self.isvanilla: # SBM Ddim reverses cond/uncond.
                output_x = torch.cat([onx, opx])
            else:
                output_x = torch.cat([opx, onx]) 
            
            # px = x
            # conp = contexts
            # opx = matsepcalc(px, conp, True, 2)
            # output_x = opx 
        self.pn = not self.pn
        self.count = 0
        # self.count += 1

        # limit = 70 if self.isxl else 16

        # if self.count == limit:
        #     self.pn = not self.pn
        #     self.count = 0
        #     self.pfirst = False
        #     self.condi += 1
        return output_x

    return forward

def split_dims(x_t, height, width, self=None):
    """Split an attention layer dimension to height + width.
    The original estimate was latent_h = sqrt(hw_ratio*x_t),
    rounding to the nearest value. However, this proved inaccurate.
    The actual operation seems to be as follows:
    - Divide h,w by 8, rounding DOWN.
    - For every new layer (of 4), divide both by 2 and round UP (then back up).
    - Multiply h*w to yield x_t.
    There is no inverse function to this set of operations,
    so instead we mimic them without the multiplication part using the original h+w.
    It's worth noting that no known checkpoints follow a different system of layering,
    but it's theoretically possible. Please report if encountered.
    """
    scale = math.ceil(math.log2(math.sqrt(height * width / x_t)))
    latent_h = repeat_div(height, scale)
    latent_w = repeat_div(width, scale)
    if x_t > latent_h * latent_w and hasattr(self, "nei_multi"):
        latent_h, latent_w = self.nei_multi[1], self.nei_multi[0] 
        while latent_h * latent_w != x_t:
            latent_h, latent_w = latent_h // 2, latent_w // 2

    return latent_h, latent_w

def repeat_div(x,y):
    """Imitates dimension halving common in convolution operations.
    
    This is a pretty big assumption of the model,
    but then if some model doesn't work like that it will be easy to spot.
    """
    while y > 0:
        x = math.ceil(x / 2)
        y = y - 1
    return x

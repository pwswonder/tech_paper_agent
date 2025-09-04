"""
services/codegen_autoblocks.v5
------------------------------
Clean module with decoder options + guards.
"""

from __future__ import annotations
from typing import Dict, Any, List


# --------------------------- Helpers ---------------------------
def _ensure_dict(d: Any) -> Dict[str, Any]:
    return d if isinstance(d, dict) else {}


def _ensure_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def _get_int(d: Dict[str, Any], key: str, default: int) -> int:
    try:
        v = d.get(key, default)
        return int(v) if v is not None else default
    except Exception:
        return default


def _get_bool(d: Dict[str, Any], key: str, default: bool) -> bool:
    try:
        v = d.get(key, default)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("1", "true", "yes", "y")
        return bool(v)
    except Exception:
        return default


def _safe_name_fmt(fmt: str, fallback: str) -> str:
    bad = ['"', "'", ";", "(", ")", "{i:"]
    if not isinstance(fmt, str) or any(ch in fmt for ch in bad) or " " in fmt:
        return fallback
    return fmt


# ----------------------- Transformer (Encoder) -----------------------
def _gen_transformer_encoder_layers_code(dims: Dict[str, Any]) -> str:
    enc = _ensure_dict(dims.get("encoder_layers"))
    count = _get_int(enc, "count", _get_int(dims, "num_layers", 6))
    name_fmt = _safe_name_fmt(enc.get("name_fmt", "enc_{i}"), "enc_{i}")
    p = _ensure_dict(enc.get("params"))
    d_model = p.get("d_model", "d_model")
    num_heads = p.get("num_heads", "num_heads")
    ffn_dim = p.get("ffn_dim", "ffn_dim")
    dropout_rate = p.get("dropout_rate", "dropout_rate")

    lines = [
        f"for i in range({count}):",
        "    x = EncoderLayer(",
        f"        d_model={d_model}, num_heads={num_heads}, ffn_dim={ffn_dim},",
        '        dropout_rate={}, name=f"{}"'.format(dropout_rate, name_fmt),
        "    )(x)",
    ]
    return "\n".join(lines)


# ----------------------- Transformer (Decoder, options) ---------------
def _gen_transformer_decoder_layers_code(dims: Dict[str, Any]) -> str:
    """
    Decoder stack without cross-attention; supports optional knobs.

    dims.decoder_layers options:
      - use_causal_mask: bool
      - attn_impl: str            # 'flash' | 'eager' | ...
      - ffn_variant: str          # 'relu' | 'gelu' | 'geglu' | 'swiGLU' | ...
      - scan: bool
      - shard: bool
      - dropout_attn: float
      - activation: str
      - attn_nb_features: int
    """
    dec = _ensure_dict(dims.get("decoder_layers"))
    count = _get_int(dec, "count", 6)
    name_fmt = _safe_name_fmt(dec.get("name_fmt", "dec_{i}"), "dec_{i}")
    # optional knobs
    use_causal_mask = _get_bool(dec, "use_causal_mask", False)
    attn_impl = dec.get("attn_impl", "")
    ffn_variant = dec.get("ffn_variant", "")
    scan_flag = _get_bool(dec, "scan", False)
    shard_flag = _get_bool(dec, "shard", False)
    dropout_attn = dec.get("dropout_attn", None)
    activation = dec.get("activation", "")
    attn_nb_features = dec.get("attn_nb_features", None)
    # params
    p = _ensure_dict(dec.get("params"))
    d_model = p.get("d_model", "d_model")
    num_heads = p.get("num_heads", "num_heads")
    ffn_dim = p.get("ffn_dim", "ffn_dim")
    dropout_rate = p.get("dropout_rate", "dropout_rate")

    lines: List[str] = []
    lines.append(f"for i in range({count}):")
    lines.append("    # Build kwargs for DecoderLayer (auto-generated)")
    lines.append("    kw = dict(")
    lines.append(
        f"        d_model={d_model}, num_heads={num_heads}, ffn_dim={ffn_dim},"
    )
    lines.append('        dropout_rate={}, name=f"{}"'.format(dropout_rate, name_fmt))
    lines.append("    )")
    if use_causal_mask:
        lines.append('    kw["causal_mask"] = True')
    if isinstance(attn_impl, str) and attn_impl:
        lines.append('    kw["attn_impl"] = {}'.format(repr(attn_impl)))
    if isinstance(ffn_variant, str) and ffn_variant:
        lines.append('    kw["ffn_variant"] = {}'.format(repr(ffn_variant)))
    if scan_flag:
        lines.append('    kw["scan"] = True')
    if shard_flag:
        lines.append('    kw["shard"] = True')
    if dropout_attn is not None:
        lines.append('    kw["dropout_attn"] = {}'.format(dropout_attn))
    if isinstance(activation, str) and activation:
        lines.append('    kw["activation"] = {}'.format(repr(activation)))
    lines.append("    try:")
    lines.append("        x = DecoderLayer(**kw)(x)")
    lines.append("    except TypeError:")
    lines.append(
        "        for _k in ('causal_mask','attn_impl','ffn_variant','scan','shard','dropout_attn','activation','attn_nb_features'):"
    )
    lines.append("            kw.pop(_k, None)")
    lines.append("        x = DecoderLayer(**kw)(x)")
    return "\n".join(lines)


# --------------- Transformer MT (Decoder with X-Attn, options) --------
def _gen_transformer_mt_decoder_layers_code(dims: Dict[str, Any]) -> str:
    """
    Decoder stack with cross-attention (expects encoder memory as `enc` or `enc_out`).

    Same optional knobs as the plain decoder above (incl. attn_nb_features).
    """
    dec = _ensure_dict(dims.get("decoder_layers"))
    count = _get_int(dec, "count", 6)
    name_fmt = _safe_name_fmt(dec.get("name_fmt", "dec_{i}"), "dec_{i}")
    # optional knobs
    use_causal_mask = _get_bool(dec, "use_causal_mask", False)
    attn_impl = dec.get("attn_impl", "")
    ffn_variant = dec.get("ffn_variant", "")
    scan_flag = _get_bool(dec, "scan", False)
    shard_flag = _get_bool(dec, "shard", False)
    dropout_attn = dec.get("dropout_attn", None)
    activation = dec.get("activation", "")
    attn_nb_features = dec.get("attn_nb_features", None)
    # params
    p = _ensure_dict(dec.get("params"))
    d_model = p.get("d_model", "d_model")
    num_heads = p.get("num_heads", "num_heads")
    ffn_dim = p.get("ffn_dim", "ffn_dim")
    dropout_rate = p.get("dropout_rate", "dropout_rate")

    lines: List[str] = []
    lines.append("# resolve encoder memory variable for cross-attention")
    lines.append(
        "enc_mem = globals().get('enc', None) or locals().get('enc', None) or globals().get('enc_out', None) or locals().get('enc_out', None)"
    )
    lines.append(
        "assert enc_mem is not None, 'Expected encoder outputs in variable `enc` or `enc_out` for MT decoder.'"
    )
    lines.append(f"for i in range({count}):")
    lines.append("    # Build kwargs for DecoderLayer (auto-generated, MT)")
    lines.append("    kw = dict(")
    lines.append(
        f"        d_model={d_model}, num_heads={num_heads}, ffn_dim={ffn_dim},"
    )
    lines.append('        dropout_rate={}, name=f"{}"'.format(dropout_rate, name_fmt))
    lines.append("    )")
    if use_causal_mask:
        lines.append('    kw["causal_mask"] = True')
    if isinstance(attn_impl, str) and attn_impl:
        lines.append('    kw["attn_impl"] = {}'.format(repr(attn_impl)))
    if isinstance(ffn_variant, str) and ffn_variant:
        lines.append('    kw["ffn_variant"] = {}'.format(repr(ffn_variant)))
    if scan_flag:
        lines.append('    kw["scan"] = True')
    if shard_flag:
        lines.append('    kw["shard"] = True')
    if dropout_attn is not None:
        lines.append('    kw["dropout_attn"] = {}'.format(dropout_attn))
    if isinstance(activation, str) and activation:
        lines.append('    kw["activation"] = {}'.format(repr(activation)))
    lines.append("    try:")
    lines.append("        x = DecoderLayer(**kw)(x, enc_mem)")
    lines.append("    except TypeError:")
    lines.append(
        "        for _k in ('causal_mask','attn_impl','ffn_variant','scan','shard','dropout_attn','activation','attn_nb_features'):"
    )
    lines.append("            kw.pop(_k, None)")
    lines.append("        x = DecoderLayer(**kw)(x, enc_mem)")
    return "\n".join(lines)


# ----------------------------- ResNet -----------------------------
def _gen_resnet_stages_code(dims: Dict[str, Any]) -> str:
    stages = _ensure_list(dims.get("stages"))
    if not stages:
        stages = [
            {"filters": 64, "blocks": 2, "stride": 1},
            {"filters": 128, "blocks": 2, "stride": 2},
            {"filters": 256, "blocks": 2, "stride": 2},
            {"filters": 512, "blocks": 2, "stride": 2},
        ]
    lines = ["# ResNet stages (auto-generated)"]
    for si, st in enumerate(stages):
        filters = int(st.get("filters", 64))
        blocks = int(st.get("blocks", 2))
        stride = int(st.get("stride", 1))
        lines.append(
            f"# -- Stage {si}: filters={filters}, blocks={blocks}, stride={stride}"
        )
        lines.append(f"for bi in range({blocks}):")
        lines.append("    s = {} if bi == 0 else 1".format(stride))
        lines.append("    residual = x")
        lines.append(
            f"    x = layers.Conv2D({filters}, 3, strides=s, padding='same', use_bias=False, name='res{si}_b{{bi}}_conv1')(x)"
        )
        lines.append(
            f"    x = layers.BatchNormalization(name='res{si}_b{{bi}}_bn1')(x)"
        )
        lines.append("    x = layers.ReLU()(x)")
        lines.append(
            f"    x = layers.Conv2D({filters}, 3, padding='same', use_bias=False, name='res{si}_b{{bi}}_conv2')(x)"
        )
        lines.append(
            f"    x = layers.BatchNormalization(name='res{si}_b{{bi}}_bn2')(x)"
        )
        lines.append("    if bi == 0 and (s != 1):")
        lines.append(
            f"        residual = layers.Conv2D({filters}, 1, strides=s, use_bias=False, name='res{si}_proj')(residual)"
        )
        lines.append(
            f"        residual = layers.BatchNormalization(name='res{si}_proj_bn')(residual)"
        )
        lines.append("    try:")
        lines.append("        x = layers.Add()([x, residual])")
        lines.append("    except Exception:")
        lines.append(
            f"        residual = layers.Conv2D({filters}, 1, use_bias=False, name='res{si}_fixproj')(residual)"
        )
        lines.append(
            f"        residual = layers.BatchNormalization(name='res{si}_fixproj_bn')(residual)"
        )
        lines.append("        x = layers.Add()([x, residual])")
        lines.append("    x = layers.ReLU()(x)")
        lines.append("")
    return "\n".join(lines)


# --------------------------- CNN Family ---------------------------
def _gen_cnn_family_stages_code(dims: Dict[str, Any]) -> str:
    stages = _ensure_list(dims.get("stages"))
    if not stages:
        stages = [
            {"filters": 32, "blocks": 2, "pool": True},
            {"filters": 64, "blocks": 2, "pool": True},
            {"filters": 128, "blocks": 2, "pool": True},
        ]
    lines = ["# CNN backbone (auto-generated)"]
    for si, st in enumerate(stages):
        f = int(st.get("filters", 32))
        b = int(st.get("blocks", 2))
        k = int(st.get("kernel", 3))
        s = int(st.get("stride", 1))
        do_pool = _get_bool(st, "pool", True)
        lines.append(
            f"# -- Stage {si}: filters={f}, blocks={b}, kernel={k}, stride={s}, pool={do_pool}"
        )
        lines.append(f"for bi in range({b}):")
        lines.append(
            f"    x = layers.Conv2D({f}, {k}, strides={s}, padding='same', activation='relu', name='conv_s{si}_b{{bi}}')(x)"
        )
        lines.append(f"    x = layers.BatchNormalization(name='bn_s{si}_b{{bi}}')(x)")
        if do_pool:
            lines.append(f"x = layers.MaxPooling2D(2, name='pool_s{si}')(x)")
        lines.append("")
    return "\n".join(lines)


# ------------------------------- U-Net ------------------------------
def _gen_unet_encoder_code(dims: Dict[str, Any]) -> str:
    enc_list = _ensure_list(dims.get("encoder_blocks"))
    if not enc_list:
        enc_list = [64, 128, 256, 512]
    lines = ["# U-Net Encoder (auto-generated)"]
    lines.append("skips_unet = []")
    for ei, ch in enumerate(enc_list):
        c = int(ch)
        lines.append(f"# -- Encoder {ei}: channels={c}")
        lines.append(
            f"x = layers.Conv2D({c}, 3, padding='same', activation='relu', name='enc{ei}_conv1')(x)"
        )
        lines.append(
            f"x = layers.Conv2D({c}, 3, padding='same', activation='relu', name='enc{ei}_conv2')(x)"
        )
        lines.append("skips_unet.append(x)")
        lines.append(f"x = layers.MaxPooling2D(2, name='enc{ei}_pool')(x)")
        lines.append("")
    return "\n".join(lines)


def _gen_unet_decoder_code(dims: Dict[str, Any]) -> str:
    dec_list = _ensure_list(dims.get("decoder_blocks"))
    if not dec_list:
        dec_list = [256, 128, 64]
    lines = ["# U-Net Decoder (auto-generated)"]
    lines.append(
        "assert 'skips_unet' in globals() or 'skips_unet' in locals(), 'encoder must run before decoder'"
    )
    for di, ch in enumerate(dec_list):
        c = int(ch)
        lines.append(f"# -- Decoder {di}: channels={c}")
        lines.append(
            f"x = layers.Conv2DTranspose({c}, 2, strides=2, padding='same', name='dec{di}_up')(x)"
        )
        lines.append(f"skip = skips_unet[-(di+1)]")
        lines.append("x = layers.Concatenate()([x, skip])")
        lines.append(
            f"x = layers.Conv2D({c}, 3, padding='same', activation='relu', name='dec{di}_conv1')(x)"
        )
        lines.append(
            f"x = layers.Conv2D({c}, 3, padding='same', activation='relu', name='dec{di}_conv2')(x)"
        )
        lines.append("")
    return "\n".join(lines)


# -------------------------------- VGG --------------------------------
def _gen_vgg_stages_code(dims: Dict[str, Any]) -> str:
    vgg = _ensure_dict(dims.get("vgg"))
    convs_per_stage = vgg.get("convs_per_stage", [2, 2, 3, 3, 3])
    channels = vgg.get("channels", [64, 128, 256, 512, 512])
    kernel = int(vgg.get("kernel", 3))
    activation = vgg.get("activation", "relu")
    use_bn = _get_bool(vgg, "use_bn", True)

    if not vgg and dims.get("stages"):
        st = _ensure_list(dims.get("stages"))
        convs_per_stage = [int(s.get("blocks", 2)) for s in st]
        channels = [int(s.get("filters", 64)) for s in st]

    n = min(len(convs_per_stage), len(channels))
    convs_per_stage = convs_per_stage[:n]
    channels = channels[:n]

    lines = ["# VGG-style backbone (auto-generated)"]
    for si in range(n):
        cnum = channels[si]
        reps = int(convs_per_stage[si])
        lines.append(f"# -- VGG Stage {si}: channels={cnum}, convs={reps}")
        for ci in range(reps):
            lines.append(
                f"x = layers.Conv2D({cnum}, {kernel}, padding='same', activation='{activation}', name='vgg_s{si}_c{ci}')(x)"
            )
            if use_bn:
                lines.append(
                    f"x = layers.BatchNormalization(name='vgg_s{si}_c{ci}_bn')(x)"
                )
        lines.append(f"x = layers.MaxPooling2D(2, name='vgg_pool_s{si}')(x)")
        lines.append("")
    return "\n".join(lines)


# ------------------------------ DenseNet ------------------------------
def _gen_densenet_stages_code(dims: Dict[str, Any]) -> str:
    dn = _ensure_dict(dims.get("densenet"))
    gr = int(dn.get("growth_rate", 32))
    blocks = dn.get("blocks_per_stage", [6, 12, 24, 16])
    init_ch = int(dn.get("init_channels", 64))
    bottleneck = _get_bool(dn, "bottleneck", True)
    compression = float(dn.get("compression", 0.5))
    use_bn = _get_bool(dn, "use_bn", True)

    in_ch = init_ch
    lines = ["# DenseNet backbone (auto-generated)"]
    for si, L in enumerate(blocks):
        lines.append(
            f"# -- Dense Block {si}: layers={int(L)}, in_channels={in_ch}, growth_rate={gr}"
        )
        lines.append(f"for li in range({int(L)}):")
        lines.append("    y = x")
        if use_bn:
            lines.append(
                f"    y = layers.BatchNormalization(name='db{si}_l{{li}}_bn1')(y)"
            )
        lines.append("    y = layers.ReLU()(y)")
        if bottleneck:
            lines.append(
                f"    y = layers.Conv2D({4*gr}, 1, padding='same', use_bias=False, name='db{si}_l{{li}}_conv1')(y)"
            )
            if use_bn:
                lines.append(
                    f"    y = layers.BatchNormalization(name='db{si}_l{{li}}_bn2')(y)"
                )
            lines.append("    y = layers.ReLU()(y)")
        lines.append(
            f"    y = layers.Conv2D({gr}, 3, padding='same', use_bias=False, name='db{si}_l{{li}}_conv3')(y)"
        )
        lines.append(
            f"    x = layers.Concatenate(name='db{si}_l{{li}}_concat')([x, y])"
        )
        in_ch += gr
        last = si == len(blocks) - 1
        if not last:
            out_ch = max(1, int(in_ch * compression))
            lines.append(f"# -- Transition {si}: compress {in_ch} -> {out_ch}")
            if use_bn:
                lines.append("x = layers.BatchNormalization(name='tr{si}_bn')(x)")
                lines.append("x = layers.ReLU()(x)")
            lines.append(
                f"x = layers.Conv2D({out_ch}, 1, padding='same', use_bias=False, name='tr{si}_conv')(x)"
            )
            lines.append(f"x = layers.AveragePooling2D(2, name='tr{si}_pool')(x)")
            in_ch = out_ch
        lines.append("")
    return "\n".join(lines)


# ---------------------------- Orchestrator ----------------------------
def autofill_custom_blocks(spec: Dict[str, Any], family: str) -> Dict[str, Any]:
    """Populate spec['custom_blocks'] based on family & dims.*"""
    if not isinstance(spec, dict):
        return spec
    dims = _ensure_dict(spec.get("dims"))
    custom = _ensure_dict(spec.get("custom_blocks"))
    fam = (family or "").lower()

    if fam in ("transformer", "transformer_ts", "transformer-family", "transformer_mt"):
        # Encoder
        if not custom.get("encoder_layers"):
            custom["encoder_layers"] = _gen_transformer_encoder_layers_code(dims)
    if fam in ("transformer", "transformer_ts", "transformer-family"):
        # Plain decoder
        if not custom.get("decoder_layers"):
            custom["decoder_layers"] = _gen_transformer_decoder_layers_code(dims)
    if fam in ("transformer_mt",):
        # MT decoder (cross-attn)
        if not custom.get("decoder_layers"):
            custom["decoder_layers"] = _gen_transformer_mt_decoder_layers_code(dims)

    elif fam in ("resnet", "resnet_family"):
        if not custom.get("stages"):
            custom["stages"] = _gen_resnet_stages_code(dims)

    elif fam in ("vgg", "vgg_family"):
        if not custom.get("stages"):
            custom["stages"] = _gen_vgg_stages_code(dims)

    elif fam in ("densenet", "densenet_family"):
        if not custom.get("stages"):
            custom["stages"] = _gen_densenet_stages_code(dims)

    elif fam in ("cnn_family", "cnn", "convnet"):
        if not custom.get("stages"):
            custom["stages"] = _gen_cnn_family_stages_code(dims)

    elif fam in ("unet", "u-net"):
        if not custom.get("encoder_blocks"):
            custom["encoder_blocks"] = _gen_unet_encoder_code(dims)
        if not custom.get("decoder_blocks"):
            custom["decoder_blocks"] = _gen_unet_decoder_code(dims)

        # NEW: Swin / Performer
    elif fam in ("swin", "swin_transformer", "swin_family"):
        if not custom.get("stages"):
            custom["stages"] = _gen_swin_stages_code(dims)

    elif fam in ("performer",):
        if not custom.get("decoder_layers"):
            custom["decoder_layers"] = _gen_performer_decoder_layers_code(dims)

    elif fam in ("performer_mt", "performer-transformer-mt"):
        if not custom.get("decoder_layers"):
            custom["decoder_layers"] = _gen_performer_mt_decoder_layers_code(dims)

    # Friendly fallbacks by dims.*
    if not custom.get("decoder_layers") and isinstance(
        dims.get("decoder_layers"), dict
    ):
        _dec = _ensure_dict(dims.get("decoder_layers"))
        if _get_bool(_dec, "use_cross_attn", False):
            custom["decoder_layers"] = _gen_transformer_mt_decoder_layers_code(dims)
        else:
            custom["decoder_layers"] = _gen_transformer_decoder_layers_code(dims)

    if not custom.get("stages") and isinstance(dims.get("vgg"), dict):
        custom["stages"] = _gen_vgg_stages_code(dims)
    if not custom.get("stages") and isinstance(dims.get("densenet"), dict):
        custom["stages"] = _gen_densenet_stages_code(dims)

    spec["dims"] = dims
    spec["custom_blocks"] = custom
    return spec


# ------------------------------ Swin (minimal) ------------------------------
def _gen_swin_stages_code(dims: Dict[str, Any]) -> str:
    """
    Minimal Swin-like stage generator using only standard Keras layers.
    It approximates windowed self-attention with DepthwiseConv2D (local mixing)
    and uses tf.roll to mimic shifted windows. No custom kernels required.

    dims.swin.* (all optional, safe defaults):
      - depths: List[int]           # number of blocks per stage, e.g., [2,2,6,2]
      - embed_dims: List[int]       # channels per stage, e.g., [96,192,384,768]
      - window_size: int            # local window (odd), default 7
      - mlp_ratio: float            # MLP width ratio, default 4.0
      - patch_merging: bool         # whether to downsample between stages, default True
    """
    sw = _ensure_dict(dims.get("swin"))
    depths = _ensure_list(sw.get("depths")) or [2, 2, 6, 2]
    embeds = _ensure_list(sw.get("embed_dims")) or [96, 192, 384, 768]
    window = int(sw.get("window_size", 7))
    mlp_ratio = float(sw.get("mlp_ratio", 4.0))
    do_merge = _get_bool(sw, "patch_merging", True)

    nstage = min(len(depths), len(embeds))
    depths, embeds = depths[:nstage], embeds[:nstage]

    lines: List[str] = []
    lines.append("# Swin-like stages (auto-generated, minimal approximation)")
    lines.append(f"window_size = {window}")
    lines.append(f"mlp_ratio = {mlp_ratio}")
    for si in range(nstage):
        ch = int(embeds[si])
        d = int(depths[si])
        lines.append(f"# -- Stage {si}: channels={ch}, depth={d}")
        # (optional) patch merging: downsample and increase channels
        if do_merge and si > 0:
            lines.append(
                f"x = layers.Conv2D({ch}, 2, strides=2, padding='same', use_bias=False, name='swin{si}_merge')(x)"
            )
            lines.append(f"x = layers.BatchNormalization(name='swin{si}_merge_bn')(x)")
        # repeated local-mixing blocks
        lines.append(f"for bi in range({d}):")
        # W-MSA approx
        lines.append("    x0 = x")
        lines.append("    x = layers.LayerNormalization(name=f'swin{si}_b{bi}_ln1')(x)")
        lines.append(
            "    x = layers.DepthwiseConv2D(window_size, padding='same', name=f'swin{si}_b{bi}_wmsa')(x)"
        )
        lines.append("    x = layers.Add(name=f'swin{si}_b{bi}_res1')([x0, x])")
        # SW-MSA approx via roll
        lines.append("    x0 = x")
        lines.append("    try:")
        lines.append("        import tensorflow as tf")
        lines.append("        x = tf.roll(x, shift=window_size//2, axis=1)")
        lines.append("        x = tf.roll(x, shift=window_size//2, axis=2)")
        lines.append("    except Exception:")
        lines.append("        pass  # if tf not present at template-eval time")
        lines.append("    x = layers.LayerNormalization(name=f'swin{si}_b{bi}_ln2')(x)")
        lines.append(
            "    x = layers.DepthwiseConv2D(window_size, padding='same', name=f'swin{si}_b{bi}_swmsa')(x)"
        )
        lines.append("    try:")
        lines.append("        x = tf.roll(x, shift=-(window_size//2), axis=1)")
        lines.append("        x = tf.roll(x, shift=-(window_size//2), axis=2)")
        lines.append("    except Exception:")
        lines.append("        pass")
        lines.append("    x = layers.Add(name=f'swin{si}_b{bi}_res2')([x0, x])")
        # MLP
        lines.append("    x0 = x")
        lines.append("    x = layers.LayerNormalization(name=f'swin{si}_b{bi}_ln3')(x)")
        lines.append(
            f"    x = layers.Conv2D(int({mlp_ratio}*{ch}), 1, activation='gelu', name=f'swin{si}_b{bi}_mlp1')(x)"
        )
        lines.append(f"    x = layers.Conv2D({ch}, 1, name=f'swin{si}_b{bi}_mlp2')(x)")
        lines.append("    x = layers.Add(name=f'swin{si}_b{bi}_res3')([x0, x])")
        lines.append("")
    return "\n".join(lines)


# ------------------------------ Performer ------------------------------
def _gen_performer_decoder_layers_code(dims: Dict[str, Any]) -> str:
    """Plain decoder with Performer attention (attn_impl='performer')."""
    dec = _ensure_dict(dims.get("decoder_layers"))
    # force performer flavor by setting attn_impl and optional nb_features default
    dec = dict(dec, attn_impl=dec.get("attn_impl", "performer"))
    dec.setdefault("attn_nb_features", 64)
    # Reuse the fully-featured decoder generator (it already supports attn_impl/nb_features)
    dims2 = dict(dims)
    dims2["decoder_layers"] = dec
    return _gen_transformer_decoder_layers_code(dims2)


def _gen_performer_mt_decoder_layers_code(dims: Dict[str, Any]) -> str:
    """MT decoder with Performer attention (attn_impl='performer')."""
    dec = _ensure_dict(dims.get("decoder_layers"))
    dec = dict(dec, attn_impl=dec.get("attn_impl", "performer"))
    dec.setdefault("attn_nb_features", 64)
    dims2 = dict(dims)
    dims2["decoder_layers"] = dec
    return _gen_transformer_mt_decoder_layers_code(dims2)

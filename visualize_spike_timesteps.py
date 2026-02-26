"""Visualize per-expert spike distribution across timesteps.

Uses PyTorch register_forward_hook to capture LIF outputs directly —
no dependency on the model's internal hook dict.

Usage:
    python visualize_spike_timesteps.py \
        -c conf/cifar100/2_512_300E_t4.yml \
        --resume /path/to/checkpoint.pth.tar \
        --image-idx 0 \
        --output-dir ./spike_vis

    # Average over N validation images:
    python visualize_spike_timesteps.py \
        -c conf/cifar100/2_512_300E_t4.yml \
        --resume /path/to/checkpoint.pth.tar \
        --num-images 100 \
        --output-dir ./spike_vis
"""

import argparse
import base64
import io
import os
import re
import yaml
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict, OrderedDict
from PIL import Image
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, MultiStepParametricLIFNode
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model
from timm.models.helpers import clean_state_dict
from module.ms_conv import MS_MLP_Expert
import model  # registers 'sdt'


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("-c", "--config", default="", type=str)

    p = argparse.ArgumentParser()
    p.add_argument("-data-dir", default="/scratch1/bkrhee/data", type=str)
    p.add_argument("--dataset", "-d", default="torch/cifar100", type=str)
    p.add_argument("--val-split", default="validation", type=str)
    p.add_argument("--model", default="sdt", type=str)
    p.add_argument("--pooling_stat", default="1111", type=str)
    p.add_argument("--spike-mode", default="lif", type=str)
    p.add_argument("--layer", default=4, type=int)
    p.add_argument("--in-channels", default=3, type=int)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--time-steps", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--mlp-ratio", type=int, default=4)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--patch-size", type=int, default=None)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--drop", type=float, default=0.0)
    p.add_argument("--drop-path", type=float, default=0.2)
    p.add_argument("--drop-block", type=float, default=None)
    p.add_argument("--crop-pct", type=float, default=None)
    p.add_argument("--mean", type=float, nargs="+", default=None)
    p.add_argument("--std", type=float, nargs="+", default=None)
    p.add_argument("--interpolation", default="", type=str)
    p.add_argument("--TET", default=False, type=bool)
    p.add_argument("--input-size", default=None, nargs=3, type=int)
    p.add_argument("--batch-size", "-b", type=int, default=64)
    p.add_argument("--val-batch-size", "-vb", type=int, default=64)
    p.add_argument("--workers", "-j", type=int, default=4)
    p.add_argument("--no-prefetcher", action="store_true", default=False)

    p.add_argument("--resume", required=True, type=str, help="checkpoint path")
    p.add_argument("--image-idx", type=int, default=0,
                    help="index of image in val set (used when --num-images is 1)")
    p.add_argument("--num-images", type=int, default=1,
                    help="number of images to average over (default: 1)")
    p.add_argument("--output-dir", default="./spike_vis", type=str)
    p.add_argument("--device", default="cuda:0", type=str)

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            p.set_defaults(**cfg)
    args = p.parse_args(remaining)
    args.config = args_config.config
    return args


def load_model_and_checkpoint(args):
    m = create_model(
        args.model,
        T=args.time_steps,
        pretrained=False,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        pooling_stat=args.pooling_stat,
        img_size_h=args.img_size,
        img_size_w=args.img_size,
        patch_size=args.patch_size,
        embed_dims=args.dim,
        mlp_ratios=args.mlp_ratio,
        in_channels=args.in_channels,
        qkv_bias=False,
        depths=args.layer,
        sr_ratios=1,
        spike_mode=args.spike_mode,
        dvs_mode=False,
        TET=args.TET,
    )

    ckpt = torch.load(args.resume, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = clean_state_dict(ckpt["state_dict"])
    else:
        state_dict = ckpt
    result = m.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"[WARN] Missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"[WARN] Unexpected keys: {result.unexpected_keys}")

    m = m.to(args.device)
    m.eval()
    return m


def make_loader(args):
    dataset_eval = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.val_split,
        is_training=False,
        batch_size=1,
    )
    data_config = resolve_data_config(vars(args), model=None)
    loader = create_loader(
        dataset_eval,
        input_size=data_config["input_size"],
        batch_size=1,
        is_training=False,
        use_prefetcher=False,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=2,
        distributed=False,
        crop_pct=data_config["crop_pct"],
        pin_memory=False,
    )
    return loader


# ────────────────────────────────────────────────────────────────────
# PyTorch forward hooks — capture LIF outputs directly
# ────────────────────────────────────────────────────────────────────

class ExpertSpikeCapture:
    """Register forward hooks on:
      1. Every LIF node inside MS_MLP_Expert  (spike tensors)
      2. Every Top2Gating module               (dispatch_tensor for slot→token mapping)

    Captured data:
        self.data["block0_expert2_fc1_lif"]  → (T, B, D, 1, Ccap)  spike tensor
        self.dispatch["block0"]              → (B, N, E, Ccap)      dispatch tensor
    """

    def __init__(self, net):
        self.data = OrderedDict()
        self.dispatch = OrderedDict()   # block_id -> dispatch_tensor
        self._handles = []
        self._block_order = []          # track which block fires in which order
        self._block_input_mean = {}     # block_id -> mean of input tensor
        self._register(net)

    def _register(self, net):
        # 1) Expert LIF nodes
        lif_pat = re.compile(
            r"block\.(\d+)\.mlp\.experts\.(\d+)\.(fc\d+_lif)$"
        )
        # 2) Top2Gating modules  (block.X.mlp.gate)
        gate_pat = re.compile(r"block\.(\d+)\.mlp\.gate$")

        for name, module in net.named_modules():
            m = lif_pat.search(name)
            if m is not None:
                block_id = int(m.group(1))
                expert_id = int(m.group(2))
                sublayer = m.group(3)
                label = f"block{block_id}_expert{expert_id}_{sublayer}"
                handle = module.register_forward_hook(self._make_lif_hook(label))
                self._handles.append(handle)
                print(f"  [hook] {label}  <-  {name}")
                continue

            m = gate_pat.search(name)
            if m is not None:
                block_id = int(m.group(1))
                label = f"block{block_id}"
                handle = module.register_forward_hook(self._make_gate_hook(label))
                self._handles.append(handle)
                print(f"  [hook] dispatch_{label}  <-  {name}")
                continue

        # 3) Block-level hooks — track processing order
        block_pat = re.compile(r"^block\.(\d+)$")
        for name, module in net.named_modules():
            m = block_pat.match(name)
            if m is not None:
                block_id = int(m.group(1))
                handle = module.register_forward_hook(
                    self._make_block_order_hook(block_id)
                )
                self._handles.append(handle)
                print(f"  [hook] block_order_{block_id}  <-  {name}")

    def _make_lif_hook(self, label):
        def hook_fn(module, input, output):
            self.data[label] = output.detach()
        return hook_fn

    def _make_gate_hook(self, label):
        def hook_fn(module, input, output):
            # Top2Gating.forward returns (dispatch_tensor, combine_tensor, loss)
            dispatch_tensor = output[0].detach()
            self.dispatch[label] = dispatch_tensor
        return hook_fn

    def _make_block_order_hook(self, block_id):
        def hook_fn(module, input, output):
            self._block_order.append(block_id)
            x_in = input[0]  # (T, B, D, H, W) — input to this block
            self._block_input_mean[block_id] = x_in.float().mean().item()
        return hook_fn

    def clear(self):
        self.data.clear()
        self.dispatch.clear()
        self._block_order.clear()
        self._block_input_mean.clear()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def collect_expert_rates(capture_data, T):
    """Convert captured spike tensors to per-expert, per-timestep firing rates.

    Returns:
        rates: dict[(block_id, sublayer)] -> np.array of shape (num_experts, T)
            Padded with last real value for experts with fewer timesteps.
        actual_t: dict[(block_id, sublayer)] -> dict[expert_id] -> int
            Number of real timesteps each expert processed.
    """
    pat = re.compile(r"^block(\d+)_expert(\d+)_(fc\d+_lif)$")
    groups = defaultdict(dict)
    actual_t = defaultdict(dict)

    for label, tensor in capture_data.items():
        m = pat.match(label)
        if m is None:
            continue
        block_id = int(m.group(1))
        expert_id = int(m.group(2))
        sublayer = m.group(3)

        t_e = tensor.shape[0]  # actual timesteps this expert processed
        rates_per_t = []
        for t in range(min(T, t_e)):
            rates_per_t.append(tensor[t].float().mean().item())
        while len(rates_per_t) < T:
            rates_per_t.append(rates_per_t[-1] if rates_per_t else 0.0)

        groups[(block_id, sublayer)][expert_id] = rates_per_t
        actual_t[(block_id, sublayer)][expert_id] = min(t_e, T)

    result = {}
    for key, expert_dict in groups.items():
        num_experts = max(expert_dict.keys()) + 1
        arr = np.zeros((num_experts, T))
        for eid, rates in expert_dict.items():
            arr[eid] = rates
        result[key] = arr
    return result, actual_t


def accumulate_rates(accumulated, new_rates):
    for key, arr in new_rates.items():
        if key in accumulated:
            accumulated[key] = accumulated[key] + arr
        else:
            accumulated[key] = arr.copy()
    return accumulated


def build_spatial_spike_maps(capture, T, H, W):
    """Map expert LIF spikes back to token spatial positions using dispatch_tensor.

    Handles experts with fewer than T timesteps: pads by repeating the last
    real timestep's spatial map.

    Returns:
        per_expert: dict[(block_id, sublayer)] -> dict[expert_id] -> np.array (T, H, W)
        combined:   dict[(block_id, sublayer)] -> np.array (T, H, W)
    """
    N = H * W
    pat = re.compile(r"^block(\d+)_expert(\d+)_(fc\d+_lif)$")

    per_expert = defaultdict(dict)   # (block, sl) -> {eid: (T, H, W)}
    combined = {}                     # (block, sl) -> (T, H, W)

    for label, spike_tensor in capture.data.items():
        m = pat.match(label)
        if m is None:
            continue
        block_id = int(m.group(1))
        eid = int(m.group(2))
        sl = m.group(3)

        dispatch_key = f"block{block_id}"
        if dispatch_key not in capture.dispatch:
            continue

        dispatch = capture.dispatch[dispatch_key]  # (B, N, E, Ccap)

        # spike_tensor: (t_e, B, D, 1, Ccap) where t_e <= T
        t_e = spike_tensor.shape[0]
        spike_rate = spike_tensor.float().mean(dim=2).squeeze(-2)  # (t_e, B, Ccap)

        expert_dispatch = dispatch[:, :, eid, :]  # (B, N, Ccap)

        # token_spike[t, b, n] = sum_c dispatch[b, n, eid, c] * spike_rate[t, b, c]
        token_spike = torch.einsum('bnc,tbc->tbn', expert_dispatch.float(), spike_rate)

        # Average over batch -> (t_e, N) -> reshape to (t_e, H, W)
        spatial = token_spike.mean(dim=1).cpu().numpy().reshape(t_e, H, W)

        # Pad to T timesteps by repeating last real timestep
        if t_e < T:
            pad = np.tile(spatial[-1:], (T - t_e, 1, 1))
            spatial = np.concatenate([spatial, pad], axis=0)

        per_expert[(block_id, sl)][eid] = spatial

    # Build combined maps (sum over experts)
    for key, eid_dict in per_expert.items():
        total = None
        for eid, arr in eid_dict.items():
            if total is None:
                total = arr.copy()
            else:
                total = total + arr
        combined[key] = total

    return per_expert, combined


# ────────────────────────────────────────────────────────────────────
# Image capture and timestep similarity
# ────────────────────────────────────────────────────────────────────

def image_to_base64(img_tensor, mean, std):
    """Denormalize a (1, C, H, W) or (C, H, W) tensor and return base64 PNG string."""
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]  # (C, H, W)
    img = img_tensor.cpu().float()
    # Denormalize per channel
    for c in range(img.shape[0]):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1).mul(255).byte()
    img_np = img.permute(1, 2, 0).numpy()  # (H, W, C)
    pil_img = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def compute_timestep_similarity(spatial_per, actual_t, T):
    """Cosine similarity between spatial spike patterns at each pair of timesteps.

    Returns:
        dict[(block_id, sublayer)] -> dict[expert_id] -> np.array (T, T)
            sim[i][j] = cosine similarity between spatial map at t=i and t=j.
            Only computed for real (non-padded) timesteps; padded entries = NaN.
    """
    result = defaultdict(dict)
    for key, eid_dict in spatial_per.items():
        for eid, arr in eid_dict.items():
            t_e = actual_t.get(key, {}).get(eid, T) if actual_t else T
            sim = np.full((T, T), np.nan)
            for ti in range(t_e):
                vi = arr[ti].ravel()
                ni = np.linalg.norm(vi)
                if ni == 0:
                    continue
                for tj in range(ti, t_e):
                    vj = arr[tj].ravel()
                    nj = np.linalg.norm(vj)
                    if nj == 0:
                        continue
                    cos = float(np.dot(vi, vj) / (ni * nj))
                    sim[ti, tj] = cos
                    sim[tj, ti] = cos
            result[key][eid] = sim
    return result


# ────────────────────────────────────────────────────────────────────
# Pure SVG visualization (no matplotlib dependency)
# ────────────────────────────────────────────────────────────────────

# Colorblind-friendly palette for timesteps
T_COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]


def _val_to_heat(val, vmin, vmax):
    """Map value to YlOrRd-like hex color."""
    if vmax <= vmin:
        t = 0.5
    else:
        t = (val - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    # YlOrRd: yellow(1,1,0.8) -> orange(1,0.55,0.15) -> red(0.7,0,0)
    if t < 0.5:
        s = t / 0.5
        r = 1.0
        g = 1.0 - 0.45 * s
        b = 0.8 - 0.65 * s
    else:
        s = (t - 0.5) / 0.5
        r = 1.0 - 0.3 * s
        g = 0.55 - 0.55 * s
        b = 0.15 - 0.15 * s
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def generate_html(rates_dict, T, output_dir,
                   spatial_per=None, spatial_comb=None, H=8, W=8,
                   actual_t=None, similarity=None, img_b64=None):
    """Generate a single HTML file with all expert spike visualizations.

    actual_t: per-expert real timestep counts.
    similarity: dict[(block,sl)] -> dict[eid] -> (T,T) cosine sim matrix.
    img_b64: base64-encoded PNG of the input image (for overlay).
    """
    keys = sorted(rates_dict.keys())
    if not keys:
        print("No expert data to visualize.")
        return

    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Expert Spike Distribution</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:20px;background:#fafafa}",
        "h1{color:#333} h2{color:#555;margin-top:40px} h3{color:#666;margin-top:25px}",
        "table{border-collapse:collapse;margin:10px 0}",
        "td,th{border:1px solid #ccc;padding:6px 12px;text-align:center}",
        "th{background:#eee}",
        ".chart-row{display:flex;flex-wrap:wrap;gap:30px}",
        ".chart-box{background:white;border:1px solid #ddd;border-radius:8px;padding:15px}",
        "svg{display:block}",
        ".legend{display:flex;gap:15px;margin:8px 0;font-size:13px}",
        ".legend-item{display:flex;align-items:center;gap:4px}",
        ".legend-swatch{width:14px;height:14px;border-radius:2px}",
        ".spatial-group{margin-bottom:30px}",
        ".spatial-grid{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start}",
        ".spatial-cell{text-align:center;font-size:11px}",
        "</style></head><body>",
        "<h1>Expert Spike Distribution Across Timesteps</h1>",
    ]

    # Legend for timestep colors
    html_parts.append("<div class='legend'>")
    for t in range(T):
        c = T_COLORS[t % len(T_COLORS)]
        html_parts.append(f"<div class='legend-item'><div class='legend-swatch' style='background:{c}'></div>t={t}</div>")
    html_parts.append("</div>")

    # ── 1) Grouped bar chart: firing rate per expert per timestep ──
    html_parts.append("<h2>1. Firing Rate per Expert per Timestep</h2>")
    html_parts.append("<div class='chart-row'>")
    for key in keys:
        bl, sl = key
        arr = rates_dict[key]
        E, _ = arr.shape
        vmax = float(arr.max()) if arr.max() > 0 else 1.0

        chart_w = 40 + E * T * 18 + E * 10
        chart_h = 220
        bar_area_h = 160
        margin_l, margin_b = 40, 40

        svg = [f"<svg width='{chart_w}' height='{chart_h}' xmlns='http://www.w3.org/2000/svg'>"]
        svg.append(f"<text x='{chart_w//2}' y='15' text-anchor='middle' font-size='13' font-weight='bold'>Block {bl} - {sl}</text>")
        # Y axis
        for i in range(5):
            val = vmax * i / 4
            y = 20 + bar_area_h - (bar_area_h * i / 4)
            svg.append(f"<line x1='{margin_l}' y1='{y}' x2='{chart_w-5}' y2='{y}' stroke='#ddd' stroke-width='0.5'/>")
            svg.append(f"<text x='{margin_l-4}' y='{y+4}' text-anchor='end' font-size='9'>{val:.2f}</text>")
        # Bars
        for e in range(E):
            group_x = margin_l + e * (T * 18 + 10)
            for t in range(T):
                val = float(arr[e, t])
                bar_h = (val / vmax) * bar_area_h if vmax > 0 else 0
                bx = group_x + t * 18
                by = 20 + bar_area_h - bar_h
                c = T_COLORS[t % len(T_COLORS)]
                svg.append(f"<rect x='{bx}' y='{by}' width='16' height='{bar_h}' fill='{c}' rx='1'/>")
                if bar_h > 12:
                    svg.append(f"<text x='{bx+8}' y='{by+12}' text-anchor='middle' font-size='7' fill='white'>{val:.3f}</text>")
            # Expert label
            label_x = group_x + (T * 18) / 2
            svg.append(f"<text x='{label_x}' y='{20 + bar_area_h + 14}' text-anchor='middle' font-size='10'>E{e}</text>")
        svg.append("</svg>")
        html_parts.append(f"<div class='chart-box'>{''.join(svg)}</div>")
    html_parts.append("</div>")

    # ── 2) Heatmap: experts x timesteps ──
    html_parts.append("<h2>2. Expert x Timestep Heatmap</h2>")
    html_parts.append("<div class='chart-row'>")
    for key in keys:
        bl, sl = key
        arr = rates_dict[key]
        E, _ = arr.shape
        vmin_h = float(arr.min())
        vmax_h = float(arr.max())
        cell_w, cell_h = 70, 40

        html_parts.append(f"<div class='chart-box'><b>Block {bl} - {sl}</b><table>")
        html_parts.append("<tr><th></th>" + "".join(f"<th>t={t}</th>" for t in range(T)) + "</tr>")
        for e in range(E):
            html_parts.append(f"<tr><th>Expert {e}</th>")
            for t in range(T):
                val = float(arr[e, t])
                bg = _val_to_heat(val, vmin_h, vmax_h)
                # Choose text color based on brightness
                txt = "white" if val > (vmin_h + vmax_h) * 0.6 else "black"
                html_parts.append(f"<td style='background:{bg};color:{txt};min-width:{cell_w}px;height:{cell_h}px'>{val:.4f}</td>")
            html_parts.append("</tr>")
        html_parts.append("</table></div>")
    html_parts.append("</div>")

    # ── 3) Cumulative curve (SVG line chart) ──
    html_parts.append("<h2>3. Cumulative Spike Fraction per Expert</h2>")
    html_parts.append("<p style='color:#888'>Dashed line = 90%. If a curve hits 0.9 by t=2, that expert needs only 2 timesteps.</p>")
    html_parts.append("<div class='chart-row'>")
    expert_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
    for key in keys:
        bl, sl = key
        arr = rates_dict[key]
        E, _ = arr.shape
        cw, ch = 300, 220
        pad_l, pad_b, pad_t = 45, 30, 30
        plot_w = cw - pad_l - 10
        plot_h = ch - pad_b - pad_t

        svg = [f"<svg width='{cw}' height='{ch}' xmlns='http://www.w3.org/2000/svg'>"]
        svg.append(f"<text x='{cw//2}' y='16' text-anchor='middle' font-size='12' font-weight='bold'>Block {bl} - {sl}</text>")
        # Grid
        for i in range(5):
            frac = i / 4
            y = pad_t + plot_h - plot_h * frac
            svg.append(f"<line x1='{pad_l}' y1='{y}' x2='{cw-10}' y2='{y}' stroke='#ddd' stroke-width='0.5'/>")
            svg.append(f"<text x='{pad_l-4}' y='{y+4}' text-anchor='end' font-size='9'>{frac:.1f}</text>")
        # 90% dashed line
        y90 = pad_t + plot_h - plot_h * 0.9
        svg.append(f"<line x1='{pad_l}' y1='{y90}' x2='{cw-10}' y2='{y90}' stroke='#999' stroke-width='1' stroke-dasharray='4,3'/>")
        svg.append(f"<text x='{cw-8}' y='{y90-3}' font-size='8' fill='#999'>90%</text>")
        # X axis labels
        for t in range(T):
            x = pad_l + (t / max(T - 1, 1)) * plot_w
            svg.append(f"<text x='{x}' y='{ch - 8}' text-anchor='middle' font-size='10'>t={t}</text>")
        # Lines per expert
        for e in range(E):
            total = float(arr[e].sum())
            if total <= 0:
                continue
            cum = list(np.cumsum(arr[e]) / total)
            points = []
            for t in range(T):
                x = pad_l + (t / max(T - 1, 1)) * plot_w
                y = pad_t + plot_h - plot_h * cum[t]
                points.append(f"{x:.1f},{y:.1f}")
            c = expert_colors[e % len(expert_colors)]
            svg.append(f"<polyline points='{' '.join(points)}' fill='none' stroke='{c}' stroke-width='2.5'/>")
            for t in range(T):
                x = pad_l + (t / max(T - 1, 1)) * plot_w
                y = pad_t + plot_h - plot_h * cum[t]
                svg.append(f"<circle cx='{x}' cy='{y}' r='4' fill='{c}'/>")
                svg.append(f"<text x='{x}' y='{y - 7}' text-anchor='middle' font-size='8'>{cum[t]:.2f}</text>")
        # Expert legend
        for e in range(E):
            lx = pad_l + e * 60
            ly = ch - 1
            c = expert_colors[e % len(expert_colors)]
            svg.append(f"<rect x='{lx}' y='{ly - 8}' width='10' height='10' fill='{c}' rx='2'/>")
            svg.append(f"<text x='{lx + 13}' y='{ly}' font-size='9'>E{e}</text>")
        svg.append("</svg>")
        html_parts.append(f"<div class='chart-box'>{''.join(svg)}</div>")
    html_parts.append("</div>")

    # ── 4) Normalized stacked bars ──
    html_parts.append("<h2>4. Timestep Share per Expert (Normalized)</h2>")
    html_parts.append("<div class='chart-row'>")
    for key in keys:
        bl, sl = key
        arr = rates_dict[key]
        E, _ = arr.shape
        bar_w = 50
        bar_full_h = 160
        cw = 50 + E * (bar_w + 15)
        ch = bar_full_h + 60

        svg = [f"<svg width='{cw}' height='{ch}' xmlns='http://www.w3.org/2000/svg'>"]
        svg.append(f"<text x='{cw//2}' y='15' text-anchor='middle' font-size='12' font-weight='bold'>Block {bl} - {sl}</text>")
        for e in range(E):
            total = float(arr[e].sum())
            if total <= 0:
                continue
            bx = 40 + e * (bar_w + 15)
            bottom_y = 25 + bar_full_h
            for t in range(T):
                frac = float(arr[e, t]) / total
                seg_h = frac * bar_full_h
                seg_y = bottom_y - seg_h
                c = T_COLORS[t % len(T_COLORS)]
                svg.append(f"<rect x='{bx}' y='{seg_y}' width='{bar_w}' height='{seg_h}' fill='{c}'/>")
                if seg_h > 12:
                    svg.append(f"<text x='{bx + bar_w//2}' y='{seg_y + seg_h//2 + 4}' text-anchor='middle' font-size='8' fill='white'>{frac:.0%}</text>")
                bottom_y = seg_y
            svg.append(f"<text x='{bx + bar_w//2}' y='{25 + bar_full_h + 15}' text-anchor='middle' font-size='10'>E{e}</text>")
        svg.append("</svg>")
        html_parts.append(f"<div class='chart-box'>{''.join(svg)}</div>")
    html_parts.append("</div>")

    # ── 5) Timestep similarity matrices ──
    if similarity:
        html_parts.append("<h2>5. Timestep Spatial Similarity (Cosine)</h2>")
        html_parts.append("<p style='color:#888'>Cosine similarity between spatial spike patterns at each timestep pair. "
                          "Values near 1.0 = same spatial pattern across timesteps (multi-step processing may be redundant). "
                          "NaN = padded timestep.</p>")
        html_parts.append("<div class='chart-row'>")
        sim_keys = sorted(similarity.keys())
        for key in sim_keys:
            bl, sl = key
            eid_dict = similarity[key]
            for eid in sorted(eid_dict.keys()):
                sim = eid_dict[eid]
                t_e = actual_t.get(key, {}).get(eid, T) if actual_t else T
                html_parts.append(f"<div class='chart-box'><b>Block {bl} - {sl} / Expert {eid}"
                                  f"{' (t_e=' + str(t_e) + ')' if t_e < T else ''}</b><table>")
                html_parts.append("<tr><th></th>" + "".join(f"<th>t={t}</th>" for t in range(T)) + "</tr>")
                for ti in range(T):
                    html_parts.append(f"<tr><th>t={ti}</th>")
                    for tj in range(T):
                        val = sim[ti, tj]
                        if np.isnan(val):
                            html_parts.append("<td style='background:#f0f0f0;color:#bbb'>-</td>")
                        else:
                            # Green (high sim) to red (low sim)
                            g = int(val * 200)
                            r = int((1 - val) * 200)
                            bg = f"rgb({r},{g},60)"
                            txt = "white" if val > 0.5 else "black"
                            html_parts.append(f"<td style='background:{bg};color:{txt}'>{val:.3f}</td>")
                    html_parts.append("</tr>")
                html_parts.append("</table></div>")
        html_parts.append("</div>")

    # ── 6) Spatial spike feature maps per token (with image overlay) ──
    if spatial_per and spatial_comb:
        html_parts.append("<h2>6. Spatial Spike Feature Maps (per token)</h2>")
        html_parts.append("<p style='color:#888'>Each grid shows firing rate at each spatial token position, "
                          "overlaid on the input image. "
                          "Rows = experts (+ combined). Columns = original image + timesteps. "
                          "Grayed-out grids = padded (repeats last real timestep).</p>")

        # Use larger cells when we have an image to overlay
        cell_px = max(20, min(32, 280 // max(H, W)))
        grid_w = W * cell_px + 2
        grid_h = H * cell_px + 18

        spatial_keys = sorted(spatial_comb.keys())
        for key in spatial_keys:
            bl, sl = key
            html_parts.append(f"<div class='spatial-group'><h3>Block {bl} - {sl}</h3>")

            eid_dict = spatial_per.get(key, {})
            eids = sorted(eid_dict.keys())
            all_vals = []
            for eid in eids:
                all_vals.append(eid_dict[eid])
            all_vals.append(spatial_comb[key])
            global_vmin = min(float(a.min()) for a in all_vals)
            global_vmax = max(float(a.max()) for a in all_vals)

            rows = [(f"Expert {eid}", eid_dict[eid], eid) for eid in eids]
            rows.append(("Combined", spatial_comb[key], None))

            for row_label, arr, row_eid in rows:
                if row_eid is not None and actual_t:
                    t_e = actual_t.get(key, {}).get(row_eid, T)
                else:
                    t_e = T
                label_extra = f" (t_e={t_e})" if t_e < T else ""
                html_parts.append(f"<div style='margin-bottom:8px'><b>{row_label}{label_extra}</b></div>")
                html_parts.append("<div class='spatial-grid'>")

                # First column: original image (only once per row, shown at token grid size)
                if img_b64:
                    svg = [f"<svg width='{grid_w}' height='{grid_h}' xmlns='http://www.w3.org/2000/svg' "
                           f"xmlns:xlink='http://www.w3.org/1999/xlink'>"]
                    svg.append(f"<text x='{grid_w//2}' y='12' text-anchor='middle' font-size='10' fill='#555'>input</text>")
                    svg.append(f"<image x='1' y='16' width='{W*cell_px}' height='{H*cell_px}' "
                               f"preserveAspectRatio='none' "
                               f"xlink:href='data:image/png;base64,{img_b64}'/>")
                    svg.append("</svg>")
                    html_parts.append(f"<div class='spatial-cell'>{''.join(svg)}</div>")

                # Timestep columns: heatmap overlaid on image
                for t in range(T):
                    is_padded = (t >= t_e)
                    grid = arr[t]
                    svg = [f"<svg width='{grid_w}' height='{grid_h}' xmlns='http://www.w3.org/2000/svg' "
                           f"xmlns:xlink='http://www.w3.org/1999/xlink'>"]
                    label_color = "#bbb" if is_padded else "#555"
                    label_text = f"t={t} (pad)" if is_padded else f"t={t}"
                    svg.append(f"<text x='{grid_w//2}' y='12' text-anchor='middle' font-size='10' fill='{label_color}'>{label_text}</text>")
                    # Draw image underneath
                    if img_b64:
                        img_op = "0.25" if is_padded else "0.5"
                        svg.append(f"<image x='1' y='16' width='{W*cell_px}' height='{H*cell_px}' "
                                   f"preserveAspectRatio='none' opacity='{img_op}' "
                                   f"xlink:href='data:image/png;base64,{img_b64}'/>")
                    # Heatmap cells on top
                    heat_op = "0.25" if is_padded else "0.65"
                    for r in range(H):
                        for c in range(W):
                            val = float(grid[r, c])
                            color = _val_to_heat(val, global_vmin, global_vmax)
                            x = c * cell_px + 1
                            y = r * cell_px + 16
                            stroke = "#999" if is_padded else "#eee"
                            dash = " stroke-dasharray='3,2'" if is_padded else ""
                            svg.append(f"<rect x='{x}' y='{y}' width='{cell_px-1}' height='{cell_px-1}' "
                                       f"fill='{color}' stroke='{stroke}' stroke-width='0.5' "
                                       f"opacity='{heat_op}'{dash}/>")
                    svg.append("</svg>")
                    html_parts.append(f"<div class='spatial-cell'>{''.join(svg)}</div>")
                html_parts.append("</div>")

            # Color scale bar
            bar_w = min((T + 1) * (grid_w + 14), 500)
            html_parts.append(f"<div style='margin-top:6px;font-size:10px;color:#888'>")
            html_parts.append(f"<svg width='{bar_w}' height='26' xmlns='http://www.w3.org/2000/svg'>")
            n_stops = 60
            stop_w = bar_w / n_stops
            for si in range(n_stops):
                frac = si / (n_stops - 1)
                val = global_vmin + frac * (global_vmax - global_vmin)
                color = _val_to_heat(val, global_vmin, global_vmax)
                html_parts.append(f"<rect x='{si * stop_w}' y='0' width='{stop_w + 1}' height='12' fill='{color}'/>")
            html_parts.append(f"<text x='0' y='22' font-size='9'>{global_vmin:.3f}</text>")
            html_parts.append(f"<text x='{bar_w}' y='22' text-anchor='end' font-size='9'>{global_vmax:.3f}</text>")
            html_parts.append("</svg></div>")

            html_parts.append("</div>")  # close spatial-group

    html_parts.append("</body></html>")

    path = os.path.join(output_dir, "expert_spikes.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"Saved: {path}")


def print_expert_table(rates_dict, T, actual_t=None):
    keys = sorted(rates_dict.keys())
    for key in keys:
        bl, sl = key
        arr = rates_dict[key]
        E = arr.shape[0]
        print(f"\n=== Block {bl} -- {sl} ===")
        header = f"  {'Expert':<12s}  {'t_e':<4s}"
        for t in range(T):
            header += f"  t={t:<6d}"
        header += f"  {'total':<8s}  {'t0/tot':<8s}  {'t01/tot':<8s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for e in range(E):
            t_e = actual_t.get(key, {}).get(e, T) if actual_t else T
            row = f"  Expert {e:<4d}  {t_e:<4d}"
            total = arr[e].sum()
            for t in range(T):
                val_str = f"{arr[e, t]:.4f}"
                if t >= t_e:
                    val_str += "*"  # mark padded timesteps
                row += f"  {val_str:<8s}"
            ratio_0 = arr[e, 0] / total if total > 0 else 0
            ratio_01 = (arr[e, 0] + arr[e, 1]) / total if total > 0 and T > 1 else ratio_0
            row += f"  {total:.4f}  {ratio_0:>6.1%}    {ratio_01:>6.1%}"
            print(row)
        if actual_t and any(actual_t.get(key, {}).get(e, T) < T for e in range(E)):
            print("  (* = padded, repeats last real timestep)")


def save_csvs(rates_dict, T, output_dir):
    """Save firing rate data as CSV files — no matplotlib needed."""
    import csv
    keys = sorted(rates_dict.keys())
    for key in keys:
        bl, sl = key
        arr = rates_dict[key]  # (E, T)
        E = arr.shape[0]
        fname = os.path.join(output_dir, f"block{bl}_{sl}.csv")
        with open(fname, "w", newline="") as f:
            w = csv.writer(f)
            header = ["expert"] + [f"t{t}" for t in range(T)] + ["total", "t0_ratio", "t01_ratio"]
            w.writerow(header)
            for e in range(E):
                total = float(arr[e].sum())
                r0 = arr[e, 0] / total if total > 0 else 0
                r01 = (arr[e, 0] + arr[e, 1]) / total if total > 0 and T > 1 else r0
                row = [f"expert{e}"] + [f"{arr[e, t]:.6f}" for t in range(T)]
                row += [f"{total:.6f}", f"{r0:.4f}", f"{r01:.4f}"]
                w.writerow(row)
        print(f"Saved: {fname}")


def _compute_token_hw(args):
    """Derive spatial token grid (H, W) from pooling_stat and img_size."""
    ratio = 1
    for c in args.pooling_stat:
        if c == "1":
            ratio *= 2
    H = W = args.img_size // ratio
    return H, W


def accumulate_spatial(accumulated, new_per_expert, new_combined):
    """Accumulate spatial spike maps across images."""
    acc_per, acc_comb = accumulated
    for key, eid_dict in new_per_expert.items():
        if key not in acc_per:
            acc_per[key] = {}
        for eid, arr in eid_dict.items():
            if eid in acc_per[key]:
                acc_per[key][eid] = acc_per[key][eid] + arr
            else:
                acc_per[key][eid] = arr.copy()
    for key, arr in new_combined.items():
        if key in acc_comb:
            acc_comb[key] = acc_comb[key] + arr
        else:
            acc_comb[key] = arr.copy()
    return acc_per, acc_comb


def save_spatial_csvs(spatial_per, spatial_comb, T, H, W, output_dir):
    """Save spatial spike maps as CSV: one file per (block, sublayer, expert/combined)."""
    import csv
    for key in sorted(spatial_comb.keys()):
        bl, sl = key
        eid_dict = spatial_per.get(key, {})
        # Combined map
        arr = spatial_comb[key]  # (T, H, W)
        fname = os.path.join(output_dir, f"spatial_block{bl}_{sl}_combined.csv")
        with open(fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestep", "row", "col", "firing_rate"])
            for t in range(T):
                for r in range(H):
                    for c in range(W):
                        w.writerow([t, r, c, f"{arr[t, r, c]:.6f}"])
        print(f"Saved: {fname}")
        # Per-expert maps
        for eid in sorted(eid_dict.keys()):
            earr = eid_dict[eid]
            fname = os.path.join(output_dir, f"spatial_block{bl}_{sl}_expert{eid}.csv")
            with open(fname, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestep", "row", "col", "firing_rate"])
                for t in range(T):
                    for r in range(H):
                        for c in range(W):
                            w.writerow([t, r, c, f"{earr[t, r, c]:.6f}"])
            print(f"Saved: {fname}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.resume} ...")
    net = load_model_and_checkpoint(args)

    # Register PyTorch forward hooks on expert LIF nodes
    print("\nRegistering forward hooks on expert LIF nodes ...")
    capture = ExpertSpikeCapture(net)
    if not capture._handles:
        print("[ERROR] No expert LIF nodes found. Dumping all module names:")
        for name, _ in net.named_modules():
            print(f"  {name}")
        return

    print(f"  → {len(capture._handles)} hooks registered.\n")

    T = args.time_steps
    N = args.num_images
    H, W = _compute_token_hw(args)
    print(f"  Token grid: H={H}, W={W}  (N={H*W} tokens)\n")
    loader = make_loader(args)

    # Resolve data_config for denormalization
    data_config = resolve_data_config(vars(args), model=None)
    img_mean = list(data_config["mean"])
    img_std = list(data_config["std"])

    print(f"Running {N} image(s) through the model ...")
    accumulated = {}
    acc_spatial = ({}, {})  # (per_expert, combined)
    actual_t = {}  # (block, sl) -> {eid: t_e}  — same for all images
    img_b64 = None  # base64 PNG of first image (for overlay)
    count = 0

    with torch.no_grad():
        for i, (img, target) in enumerate(loader):
            if N == 1 and i < args.image_idx:
                continue

            img = img.float().to(args.device)
            target = target.to(args.device)

            functional.reset_net(net)
            capture.clear()

            output, _ = net(img, hook=dict())

            rates, img_actual_t = collect_expert_rates(capture.data, T)
            accumulated = accumulate_rates(accumulated, rates)
            if not actual_t:
                actual_t = img_actual_t  # model property, same across images

            # Spatial maps
            per_exp, comb = build_spatial_spike_maps(capture, T, H, W)
            acc_spatial = accumulate_spatial(acc_spatial, per_exp, comb)
            count += 1

            if count == 1:
                # Capture first image for overlay
                img_b64 = image_to_base64(img, img_mean, img_std)
                pred = output.argmax(dim=-1).item()
                gt = target.item()
                print(f"  Image {i}: pred={pred}, gt={gt}")
                print(f"  Captured {len(capture.data)} spike tensors, "
                      f"{len(capture.dispatch)} dispatch tensors")
                for label, tensor in capture.data.items():
                    print(f"    {label:40s}  shape={list(tensor.shape)}  "
                          f"(t_e={tensor.shape[0]})")
                for label, tensor in capture.dispatch.items():
                    print(f"    dispatch_{label:36s}  shape={list(tensor.shape)}")

                # Diagnostic: show expert_timesteps from the MoE modules
                for bname, bmodule in net.named_modules():
                    if hasattr(bmodule, 'expert_timesteps'):
                        print(f"  [diag] {bname}.expert_timesteps = "
                              f"{bmodule.expert_timesteps}")

                # Diagnostic: block processing order & input stats
                print(f"\n  [diag] Block processing order: {capture._block_order}")
                for bid in sorted(capture._block_input_mean.keys()):
                    print(f"  [diag] Block {bid} input mean: "
                          f"{capture._block_input_mean[bid]:.6f}")

                # Diagnostic: verify hook data insertion order matches block order
                print(f"\n  [diag] Hook data insertion order:")
                for label in capture.data.keys():
                    print(f"    {label}")
                print(f"  [diag] Dispatch insertion order:")
                for label in capture.dispatch.keys():
                    print(f"    {label}")

            if count >= N:
                break

    capture.remove_hooks()

    if count == 0:
        print("No images processed.")
        return

    for key in accumulated:
        accumulated[key] /= count
    # Average spatial maps
    spatial_per, spatial_comb = acc_spatial
    for key in spatial_per:
        for eid in spatial_per[key]:
            spatial_per[key][eid] /= count
    for key in spatial_comb:
        spatial_comb[key] /= count

    print(f"\nAveraged over {count} image(s).")
    print(f"Found expert data for {len(accumulated)} (block, sublayer) groups.")
    print(f"Found spatial maps for {len(spatial_comb)} groups.\n")

    # Per-block summary for cross-checking
    print("=== Per-Block Firing Rate Summary ===")
    for key in sorted(accumulated.keys()):
        bl, sl = key
        arr = accumulated[key]
        E = arr.shape[0]
        for e in range(E):
            t_e = actual_t.get(key, {}).get(e, T)
            print(f"  Block {bl} - {sl} - Expert {e} (t_e={t_e}): "
                  f"mean={arr[e].mean():.6f}  per_t={[f'{v:.4f}' for v in arr[e]]}")
    print()

    print_expert_table(accumulated, T, actual_t)

    # Compute timestep similarity
    similarity = compute_timestep_similarity(spatial_per, actual_t, T)

    # Save CSVs
    save_csvs(accumulated, T, args.output_dir)
    if spatial_comb:
        save_spatial_csvs(spatial_per, spatial_comb, T, H, W, args.output_dir)

    # Generate HTML visualization (pure SVG, no matplotlib)
    generate_html(accumulated, T, args.output_dir,
                  spatial_per=spatial_per, spatial_comb=spatial_comb,
                  H=H, W=W, actual_t=actual_t,
                  similarity=similarity, img_b64=img_b64)

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

"""Compute Temporal Information Concentration (TIC) per expert.

Based on: "Exploring Temporal Information Dynamics in Spiking Neural Networks"
(AAAI 2023, Shen et al.)

Key formulas:
    I_t = (1/N) * sum_n || grad_theta log f(y_n | x_{i<=t}^n) ||^2   (Eq. 6)
    IC  = sum_t (t * I_t) / sum_t (I_t)                               (Eq. 7)

For each validation sample, we:
  1. Forward the model to get per-timestep logits (T, B, num_classes)
  2. For each timestep t = 1..T:
       loss_t = CrossEntropy(logits[t-1], label)
       -- In an SNN, logits[t] already encodes accumulated temporal
          information via LIF membrane potential dynamics.
       Backprop loss_t, collect ||grad||^2 per expert parameter group
       -- Per-expert gradients only counted at t <= t_e (the expert's
          processing timesteps). Beyond t_e the output is just a repeat
          via expand, and gradient accumulation through it is artificial.
  3. Average over N samples
  4. Compute IC per expert

Usage:
    python compute_tic.py 
        -c conf/conf.yml
        --resume /checkpoint.pth.tar
        --num-images 100 
        --output-dir ./tic_vis

"""

import argparse
import os
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict
from spikingjelly.clock_driven import functional
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model
from timm.models.helpers import clean_state_dict
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
import model  # registers 'sdt'
import dvs_utils


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    p.add_argument("--num-images", type=int, default=100,
                    help="number of validation images to average over")
    p.add_argument("--output-dir", default="./tic_vis", type=str)
    p.add_argument("--device", default="cuda:0", type=str)

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            p.set_defaults(**cfg)
    args = p.parse_args(remaining)
    args.config = args_config.config
    return args


# ---------------------------------------------------------------------------
# Model / data helpers (same as visualize_spike_timesteps.py)
# ---------------------------------------------------------------------------

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
        dvs_mode=args.dataset in dvs_utils.DVS_DATASET,
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
    if args.dataset in dvs_utils.DVS_DATASET:
        if args.dataset == "gesture":
            # dataset_eval = DVS128Gesture(
            #     args.data_dir, train=False, data_type="frame",
            #     frames_number=args.time_steps, split_by="number",
            #     transform=dvs_utils.SpatioTemporalDenoise(temporal_window=0, spatial_kernel=3, threshold=3),
            # )
            dataset_eval = DVS128Gesture(
                args.data_dir, train=False, data_type="frame",
                frames_number=args.time_steps, split_by="number",
            )
        elif args.dataset == "cifar10-dvs":
            ds = CIFAR10DVS(
                args.data_dir, data_type="frame",
                frames_number=args.time_steps, split_by="number",
                transform=dvs_utils.Resize(64),
            )
            _, dataset_eval = dvs_utils.split_to_train_test_set(0.9, ds, 10)
        elif args.dataset == "cifar10-dvs-tet":
            import os as _os
            dataset_eval = dvs_utils.DVSCifar10(
                root=_os.path.join(args.data_dir, "test"), train=False)
        else:
            raise ValueError(f"Unknown DVS dataset: {args.dataset}")
        return torch.utils.data.DataLoader(
            dataset_eval, batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=False,
        )

    # Standard image datasets via timm
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


# ---------------------------------------------------------------------------
# Parameter group identification
# ---------------------------------------------------------------------------

def get_expert_param_groups(mdl):
    """Return {(layer_idx, expert_id): [params]} for all expert parameters."""
    groups = {}
    for name, param in mdl.named_parameters():
        # e.g. block.0.mlp.experts.2.fc1_conv.weight
        parts = name.split(".")
        if "experts" in parts:
            layer_idx = int(parts[1])           # block.<layer>
            expert_idx = int(parts[parts.index("experts") + 1])
            key = (layer_idx, expert_idx)
            groups.setdefault(key, []).append(param)
    return groups


def get_gate_param_groups(mdl):
    """Return {layer_idx: [params]} for gate/router parameters."""
    groups = {}
    for name, param in mdl.named_parameters():
        if "gate" in name:
            parts = name.split(".")
            layer_idx = int(parts[1])
            groups.setdefault(layer_idx, []).append(param)
    return groups


def get_attn_param_groups(mdl):
    """Return {layer_idx: [params]} for attention parameters."""
    groups = {}
    for name, param in mdl.named_parameters():
        if "attn" in name:
            parts = name.split(".")
            layer_idx = int(parts[1])
            groups.setdefault(layer_idx, []).append(param)
    return groups


def get_expert_timesteps(mdl):
    """Extract expert_timesteps from first MoE block."""
    for module in mdl.modules():
        if hasattr(module, "expert_timesteps"):
            return list(module.expert_timesteps)
    return None


def get_token_counts_per_expert(mdl):
    """After a forward pass, read gate.last_masks from each MoE block
    to count how many tokens were routed to each expert.

    Returns:
        {(layer_idx, expert_id): int}
    """
    counts = {}
    blocks = getattr(mdl, "block")
    for layer_idx, blk in enumerate(blocks):
        gate = blk.mlp.gate
        if gate.last_masks is None:
            continue
        # last_masks is a list of top_k masks, each (B, N, E)
        num_experts = gate.num_gates
        for eid in range(num_experts):
            total = sum(
                mask[:, :, eid].sum().item() for mask in gate.last_masks
            )
            counts[(layer_idx, eid)] = total
    return counts


# ---------------------------------------------------------------------------
# Forward without temporal average
# ---------------------------------------------------------------------------

def forward_per_timestep(mdl, x):
    """Forward pass returning per-timestep logits (T, B, num_classes)."""
    if len(x.shape) < 5:
        x = (x.unsqueeze(0)).repeat(mdl.T, 1, 1, 1, 1)
    else:
        x = x.transpose(0, 1).contiguous()

    x, _ = mdl.forward_features(x)
    x = mdl.head_lif(x)
    x = mdl.head(x)       # (T, B, num_classes)
    return x


# ---------------------------------------------------------------------------
# Core TIC computation
# ---------------------------------------------------------------------------

def compute_fisher_info(mdl, loader, args):
    """Compute per-timestep Fisher Information for experts, gates, attn, global.

    Follows the approach from Shen et al. (AAAI 2023): for each timestep t,
    reset the network, do a fresh forward pass, compute loss on the
    accumulated average output mean(logits[0:t]), then collect ||grad||^2.

    For per-expert Fisher info, only collects at t <= t_e (the expert's
    actual processing timesteps).  Beyond t_e the expert output is a repeat
    via torch.expand, and gradient accumulation through it is an artifact
    of the shared autograd graph -- not genuine temporal information.

    Returns:
        fisher_expert: {(layer, expert_id): np.array of shape (T,)}
        fisher_gate:   {layer: np.array of shape (T,)}
        fisher_attn:   {layer: np.array of shape (T,)}
        fisher_global: np.array of shape (T,)
        n_samples: int
    """
    T = args.time_steps
    device = args.device

    expert_groups = get_expert_param_groups(mdl)
    gate_groups = get_gate_param_groups(mdl)
    attn_groups = get_attn_param_groups(mdl)

    expert_timesteps = get_expert_timesteps(mdl)

    fisher_expert = {k: np.zeros(T) for k in expert_groups}
    fisher_gate = {k: np.zeros(T) for k in gate_groups}
    fisher_attn = {k: np.zeros(T) for k in attn_groups}
    fisher_global = np.zeros(T)

    n_samples = 0

    mdl.eval()
    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx >= args.num_images:
            break

        images = images.float().to(device)
        labels = labels.to(device)

        for t in range(1, T + 1):
            functional.reset_net(mdl)

            with torch.enable_grad():
                logits = forward_per_timestep(mdl, images)  # (T, B, C)
                # Accumulated average of outputs from timestep 1..t
                acc_logits = logits[:t].mean(dim=0)  # (B, num_classes)
                loss_t = F.cross_entropy(acc_logits, labels)

                mdl.zero_grad()
                loss_t.backward()

                # Capture token counts from this forward pass
                token_counts = get_token_counts_per_expert(mdl)

                # Per-expert: only count at t <= t_e to avoid gradient
                # leaking through the expand/repeat used for padding.
                # Normalize by token count so experts are comparable
                # regardless of routing frequency.
                for (layer, eid), params in expert_groups.items():
                    t_e = expert_timesteps[eid] if expert_timesteps else T
                    if t <= t_e:
                        gn = sum(
                            p.grad.pow(2).sum().item()
                            for p in params if p.grad is not None
                        )
                        n_tok = token_counts.get((layer, eid), 0)
                        if n_tok > 0:
                            gn /= n_tok
                        fisher_expert[(layer, eid)][t - 1] += gn
                    # else: leave as 0 -- no new computation from this expert

                # Per-gate (no expand issue, collect at all timesteps)
                for key, params in gate_groups.items():
                    gn = sum(
                        p.grad.pow(2).sum().item()
                        for p in params if p.grad is not None
                    )
                    fisher_gate[key][t - 1] += gn

                # Per-attention (no expand issue)
                for key, params in attn_groups.items():
                    gn = sum(
                        p.grad.pow(2).sum().item()
                        for p in params if p.grad is not None
                    )
                    fisher_attn[key][t - 1] += gn

                # Global
                gn = sum(
                    p.grad.pow(2).sum().item()
                    for p in mdl.parameters() if p.grad is not None
                )
                fisher_global[t - 1] += gn

        n_samples += 1
        if (batch_idx + 1) % 10 == 0:
            print(f"  [{batch_idx + 1}/{args.num_images}] images processed")

    # Average over samples
    if n_samples > 0:
        for k in fisher_expert:
            fisher_expert[k] /= n_samples
        for k in fisher_gate:
            fisher_gate[k] /= n_samples
        for k in fisher_attn:
            fisher_attn[k] /= n_samples
        fisher_global /= n_samples

    return fisher_expert, fisher_gate, fisher_attn, fisher_global, n_samples


def compute_ic(fisher):
    """Information Centroid: IC = sum(t * I_t) / sum(I_t).  t is 1-indexed."""
    total = fisher.sum()
    if total < 1e-15:
        return float("nan")
    timesteps = np.arange(1, len(fisher) + 1, dtype=np.float64)
    return float((timesteps * fisher).sum() / total)


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_results(fisher_expert, fisher_gate, fisher_attn, fisher_global,
                  T, expert_timesteps, n_samples):
    print(f"\n{'='*70}")
    print(f"  Temporal Information Concentration (TIC)  --  N={n_samples} images")
    print(f"{'='*70}")

    if expert_timesteps:
        print(f"  expert_timesteps = {expert_timesteps}")
    print()

    # Header
    ts_hdr = "".join(f"{'t=' + str(t+1):>12}" for t in range(T))
    print(f"  {'Component':<28}{ts_hdr}{'IC':>10}")
    print(f"  {'-'*28}{'-'*12*T}{'-'*10}")

    # Per-expert
    layers = sorted(set(l for l, _ in fisher_expert))
    for layer in layers:
        experts_in_layer = sorted(e for (l, e) in fisher_expert if l == layer)
        for eid in experts_in_layer:
            fi = fisher_expert[(layer, eid)]
            ic = compute_ic(fi)
            te = expert_timesteps[eid] if expert_timesteps else T
            label = f"L{layer} Expert{eid} (t_e={te})"
            # Mark timesteps beyond t_e as "-" (not computed)
            vals = ""
            for t_idx in range(T):
                if t_idx < te:
                    vals += f"{fi[t_idx]:12.4e}"
                else:
                    vals += f"{'--':>12}"
            print(f"  {label:<28}{vals}{ic:10.3f}")
        # Gate
        if layer in fisher_gate:
            fi = fisher_gate[layer]
            ic = compute_ic(fi)
            vals = "".join(f"{v:12.4e}" for v in fi)
            print(f"  {'L' + str(layer) + ' Gate':<28}{vals}{ic:10.3f}")
        # Attention
        if layer in fisher_attn:
            fi = fisher_attn[layer]
            ic = compute_ic(fi)
            vals = "".join(f"{v:12.4e}" for v in fi)
            print(f"  {'L' + str(layer) + ' Attention':<28}{vals}{ic:10.3f}")
        print()

    # Global
    ic_g = compute_ic(fisher_global)
    vals = "".join(f"{v:12.4e}" for v in fisher_global)
    print(f"  {'GLOBAL':<28}{vals}{ic_g:10.3f}")
    print()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csvs(fisher_expert, fisher_gate, fisher_attn, fisher_global,
              T, expert_timesteps, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Expert CSV
    path = os.path.join(output_dir, "tic_expert.csv")
    with open(path, "w", encoding="utf-8") as f:
        hdr = "layer,expert,t_e," + ",".join(f"I_{t+1}" for t in range(T)) + ",IC\n"
        f.write(hdr)
        for (layer, eid) in sorted(fisher_expert):
            fi = fisher_expert[(layer, eid)]
            ic = compute_ic(fi)
            te = expert_timesteps[eid] if expert_timesteps else ""
            vals = ",".join(f"{v:.6e}" for v in fi)
            f.write(f"{layer},{eid},{te},{vals},{ic:.4f}\n")
    print(f"  Saved {path}")

    # Gate CSV
    path = os.path.join(output_dir, "tic_gate.csv")
    with open(path, "w", encoding="utf-8") as f:
        hdr = "layer," + ",".join(f"I_{t+1}" for t in range(T)) + ",IC\n"
        f.write(hdr)
        for layer in sorted(fisher_gate):
            fi = fisher_gate[layer]
            ic = compute_ic(fi)
            vals = ",".join(f"{v:.6e}" for v in fi)
            f.write(f"{layer},{vals},{ic:.4f}\n")
    print(f"  Saved {path}")

    # Global CSV
    path = os.path.join(output_dir, "tic_global.csv")
    with open(path, "w", encoding="utf-8") as f:
        hdr = ",".join(f"I_{t+1}" for t in range(T)) + ",IC\n"
        f.write(hdr)
        ic = compute_ic(fisher_global)
        vals = ",".join(f"{v:.6e}" for v in fisher_global)
        f.write(f"{vals},{ic:.4f}\n")
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# HTML / SVG visualization
# ---------------------------------------------------------------------------

def _color_for_ic(ic, T):
    """Blue (early) -> Red (late) based on IC relative to T."""
    if np.isnan(ic):
        return "#999"
    frac = (ic - 1.0) / max(T - 1.0, 1.0)   # 0 = earliest, 1 = latest
    frac = max(0.0, min(1.0, frac))
    r = int(40 + 215 * frac)
    b = int(215 - 215 * frac + 40)
    g = int(80 * (1.0 - abs(frac - 0.5) * 2))
    return f"#{r:02x}{g:02x}{b:02x}"


EXPERT_COLORS = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63"]


def generate_html(fisher_expert, fisher_gate, fisher_attn, fisher_global,
                  T, expert_timesteps, output_dir, n_samples):
    layers = sorted(set(l for l, _ in fisher_expert))
    num_experts = max(e for _, e in fisher_expert) + 1 if fisher_expert else 0

    parts = []
    parts.append("""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>TIC Analysis</title>
<style>
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #f5f5f5;
       margin: 0; padding: 20px; color: #333; }
h1 { text-align: center; }
h2 { margin-top: 40px; border-bottom: 2px solid #ccc; padding-bottom: 6px; }
.card { background: #fff; border-radius: 8px; padding: 20px; margin: 16px 0;
        box-shadow: 0 1px 4px rgba(0,0,0,.12); }
table { border-collapse: collapse; margin: 12px 0; }
th, td { padding: 6px 14px; border: 1px solid #ddd; text-align: center; }
th { background: #f0f0f0; }
.meta { color: #666; font-size: 0.9em; }
svg text { font-family: 'Segoe UI', system-ui, sans-serif; }
</style></head><body>
""")

    parts.append(f"<h1>Temporal Information Concentration (TIC)</h1>")
    parts.append(f'<p class="meta" style="text-align:center">'
                 f'N={n_samples} images, T={T} timesteps')
    if expert_timesteps:
        parts.append(f', expert_timesteps={expert_timesteps}')
    parts.append("</p>")

    # =================================================================
    # Section 1: Fisher Information bar charts per layer
    # =================================================================
    parts.append("<h2>1. Fisher Information per Timestep</h2>")

    for layer in layers:
        experts_in_layer = sorted(e for (l, e) in fisher_expert if l == layer)
        E = len(experts_in_layer)

        # Find max value for scaling
        max_val = max(
            fisher_expert[(layer, e)].max() for e in experts_in_layer
        )
        if max_val < 1e-15:
            max_val = 1.0

        chart_w, chart_h = 500, 220
        margin_l, margin_b, margin_t = 70, 40, 30
        plot_w = chart_w - margin_l - 20
        plot_h = chart_h - margin_b - margin_t
        bar_group_w = plot_w / T
        bar_w = bar_group_w / (E + 1)

        svg = [f'<div class="card"><h3>Layer {layer}</h3>']
        svg.append(f'<svg width="{chart_w}" height="{chart_h}" '
                   f'style="overflow:visible">')

        # Y-axis
        for i in range(5):
            y = margin_t + plot_h - (plot_h * i / 4)
            v = max_val * i / 4
            svg.append(f'<line x1="{margin_l}" y1="{y}" x2="{margin_l + plot_w}" '
                       f'y2="{y}" stroke="#e0e0e0" stroke-width="0.5"/>')
            svg.append(f'<text x="{margin_l - 6}" y="{y + 4}" text-anchor="end" '
                       f'font-size="10" fill="#666">{v:.1e}</text>')

        # Bars
        for ti in range(T):
            gx = margin_l + ti * bar_group_w
            for idx, eid in enumerate(experts_in_layer):
                te = expert_timesteps[eid] if expert_timesteps else T
                val = fisher_expert[(layer, eid)][ti]
                h = (val / max_val) * plot_h if max_val > 0 else 0
                x = gx + (idx + 0.5) * bar_w
                y = margin_t + plot_h - h
                color = EXPERT_COLORS[eid % len(EXPERT_COLORS)]
                # Dim bars beyond t_e (no data there)
                opacity = "0.85" if ti < te else "0.15"
                svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.85:.1f}" '
                           f'height="{h:.1f}" fill="{color}" opacity="{opacity}"/>')

            # X label
            cx = gx + bar_group_w / 2
            svg.append(f'<text x="{cx:.1f}" y="{margin_t + plot_h + 16}" '
                       f'text-anchor="middle" font-size="11">t={ti + 1}</text>')

        # Legend
        lx = margin_l + plot_w + 10
        for idx, eid in enumerate(experts_in_layer):
            te = expert_timesteps[eid] if expert_timesteps else "?"
            color = EXPERT_COLORS[eid % len(EXPERT_COLORS)]
            ly = margin_t + idx * 18
            svg.append(f'<rect x="{lx}" y="{ly}" width="12" height="12" '
                       f'fill="{color}"/>')
            svg.append(f'<text x="{lx + 16}" y="{ly + 10}" font-size="10">'
                       f'E{eid} (t_e={te})</text>')

        svg.append("</svg></div>")
        parts.append("\n".join(svg))

    # =================================================================
    # Section 2: Information Centroid (IC) summary table
    # =================================================================
    parts.append("<h2>2. Information Centroid (IC)</h2>")
    parts.append('<div class="card">')
    parts.append('<p>IC = 1.0 means all info at t=1 (early). '
                 'For experts, IC is computed over active timesteps only '
                 '(t=1..t_e).</p>')

    parts.append("<table><tr><th></th>")
    for layer in layers:
        parts.append(f"<th>Layer {layer}</th>")
    parts.append("</tr>")

    for eid in range(num_experts):
        te = expert_timesteps[eid] if expert_timesteps else T
        parts.append(f"<tr><td><b>Expert {eid}</b> (t_e={te})</td>")
        for layer in layers:
            fi = fisher_expert.get((layer, eid))
            if fi is not None:
                ic = compute_ic(fi)
                # Color relative to this expert's t_e range
                color = _color_for_ic(ic, te)
                if te == 1:
                    parts.append(f'<td style="background:{color};color:#fff;'
                                 f'font-weight:bold">{ic:.3f} (trivial)</td>')
                else:
                    parts.append(f'<td style="background:{color};color:#fff;'
                                 f'font-weight:bold">{ic:.3f}</td>')
            else:
                parts.append("<td>-</td>")
        parts.append("</tr>")

    # Gate row
    parts.append("<tr><td><b>Gate</b></td>")
    for layer in layers:
        fi = fisher_gate.get(layer)
        if fi is not None:
            ic = compute_ic(fi)
            color = _color_for_ic(ic, T)
            parts.append(f'<td style="background:{color};color:#fff">'
                         f'{ic:.3f}</td>')
        else:
            parts.append("<td>-</td>")
    parts.append("</tr>")

    # Attention row
    parts.append("<tr><td><b>Attention</b></td>")
    for layer in layers:
        fi = fisher_attn.get(layer)
        if fi is not None:
            ic = compute_ic(fi)
            color = _color_for_ic(ic, T)
            parts.append(f'<td style="background:{color};color:#fff">'
                         f'{ic:.3f}</td>')
        else:
            parts.append("<td>-</td>")
    parts.append("</tr>")

    # Global row
    ic_g = compute_ic(fisher_global)
    color_g = _color_for_ic(ic_g, T)
    parts.append(f'<tr><td><b>GLOBAL</b></td>'
                 f'<td colspan="{len(layers)}" '
                 f'style="background:{color_g};color:#fff;font-weight:bold">'
                 f'{ic_g:.3f}</td></tr>')

    parts.append("</table></div>")

    # =================================================================
    # Section 3: Normalized Fisher distribution (fraction at each t)
    # =================================================================
    parts.append("<h2>3. Normalized Fisher Distribution per Expert</h2>")
    parts.append('<p class="meta">Shows what fraction of total Fisher '
                 'information falls at each accumulated timestep.</p>')

    for layer in layers:
        experts_in_layer = sorted(e for (l, e) in fisher_expert if l == layer)
        chart_w, chart_h = 500, 200
        margin_l, margin_b, margin_t = 70, 40, 30
        plot_w = chart_w - margin_l - 20
        plot_h = chart_h - margin_b - margin_t

        svg = [f'<div class="card"><h3>Layer {layer}</h3>']
        svg.append(f'<svg width="{chart_w}" height="{chart_h}" '
                   f'style="overflow:visible">')

        # Horizontal gridlines at 0%, 25%, 50%, 75%, 100%
        for i in range(5):
            y = margin_t + plot_h - (plot_h * i / 4)
            svg.append(f'<line x1="{margin_l}" y1="{y}" x2="{margin_l + plot_w}" '
                       f'y2="{y}" stroke="#e0e0e0" stroke-width="0.5"/>')
            svg.append(f'<text x="{margin_l - 6}" y="{y + 4}" text-anchor="end" '
                       f'font-size="10" fill="#666">{25 * i}%</text>')

        # Line plot per expert (only active timesteps t <= t_e)
        for eid in experts_in_layer:
            fi = fisher_expert[(layer, eid)]
            te = expert_timesteps[eid] if expert_timesteps else T
            total = fi[:te].sum()
            if total < 1e-15:
                continue
            frac = fi[:te] / total
            color = EXPERT_COLORS[eid % len(EXPERT_COLORS)]
            points = []
            for ti in range(te):
                x = margin_l + (ti + 0.5) * (plot_w / T)
                y = margin_t + plot_h - frac[ti] * plot_h
                points.append(f"{x:.1f},{y:.1f}")
            svg.append(f'<polyline points="{" ".join(points)}" fill="none" '
                       f'stroke="{color}" stroke-width="2.5"/>')
            # Dots
            for ti in range(te):
                x = margin_l + (ti + 0.5) * (plot_w / T)
                y = margin_t + plot_h - frac[ti] * plot_h
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" '
                           f'fill="{color}"/>')
                svg.append(f'<text x="{x:.1f}" y="{y - 8:.1f}" text-anchor="middle" '
                           f'font-size="9" fill="{color}">{frac[ti]*100:.1f}%</text>')

        # X labels
        for ti in range(T):
            cx = margin_l + (ti + 0.5) * (plot_w / T)
            svg.append(f'<text x="{cx:.1f}" y="{margin_t + plot_h + 16}" '
                       f'text-anchor="middle" font-size="11">t={ti + 1}</text>')

        # Legend
        lx = margin_l + plot_w + 10
        for idx, eid in enumerate(experts_in_layer):
            te = expert_timesteps[eid] if expert_timesteps else "?"
            color = EXPERT_COLORS[eid % len(EXPERT_COLORS)]
            ly = margin_t + idx * 18
            svg.append(f'<rect x="{lx}" y="{ly}" width="12" height="12" '
                       f'fill="{color}"/>')
            svg.append(f'<text x="{lx + 16}" y="{ly + 10}" font-size="10">'
                       f'E{eid} (t_e={te})</text>')

        svg.append("</svg></div>")
        parts.append("\n".join(svg))

    # =================================================================
    # Section 4: IC comparison bar chart
    # =================================================================
    parts.append("<h2>4. IC Comparison Across Experts</h2>")

    for layer in layers:
        experts_in_layer = sorted(e for (l, e) in fisher_expert if l == layer)
        E = len(experts_in_layer)

        chart_w, chart_h = 400, 200
        margin_l, margin_b, margin_t = 70, 50, 30
        plot_w = chart_w - margin_l - 20
        plot_h = chart_h - margin_b - margin_t
        bar_w = plot_w / (E + 1)

        svg = [f'<div class="card"><h3>Layer {layer}</h3>']
        svg.append(f'<svg width="{chart_w}" height="{chart_h}" '
                   f'style="overflow:visible">')

        # Y-axis: IC from 1 to T
        for i in range(T):
            y = margin_t + plot_h - (plot_h * i / (T - 1)) if T > 1 else margin_t + plot_h / 2
            svg.append(f'<line x1="{margin_l}" y1="{y}" x2="{margin_l + plot_w}" '
                       f'y2="{y}" stroke="#e0e0e0" stroke-width="0.5"/>')
            svg.append(f'<text x="{margin_l - 6}" y="{y + 4}" text-anchor="end" '
                       f'font-size="10" fill="#666">{i + 1:.0f}</text>')

        # Midline (uniform IC = (T+1)/2)
        uniform_ic = (T + 1) / 2.0
        y_uni = margin_t + plot_h - ((uniform_ic - 1) / max(T - 1, 1)) * plot_h
        svg.append(f'<line x1="{margin_l}" y1="{y_uni}" x2="{margin_l + plot_w}" '
                   f'y2="{y_uni}" stroke="#999" stroke-width="1" '
                   f'stroke-dasharray="4,3"/>')
        svg.append(f'<text x="{margin_l + plot_w + 4}" y="{y_uni + 4}" '
                   f'font-size="9" fill="#999">uniform</text>')

        for idx, eid in enumerate(experts_in_layer):
            ic = compute_ic(fisher_expert[(layer, eid)])
            te = expert_timesteps[eid] if expert_timesteps else T
            if np.isnan(ic):
                continue
            x = margin_l + (idx + 0.5) * bar_w
            h = ((ic - 1) / max(T - 1, 1)) * plot_h
            y = margin_t + plot_h - h
            color = _color_for_ic(ic, T)
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.7:.1f}" '
                       f'height="{max(h, 2):.1f}" fill="{color}" rx="3"/>')
            label = f"{ic:.2f}"
            if te == 1:
                label += "*"
            svg.append(f'<text x="{x + bar_w * 0.35:.1f}" y="{y - 6:.1f}" '
                       f'text-anchor="middle" font-size="11" font-weight="bold" '
                       f'fill="{color}">{label}</text>')
            svg.append(f'<text x="{x + bar_w * 0.35:.1f}" '
                       f'y="{margin_t + plot_h + 16}" text-anchor="middle" '
                       f'font-size="10">E{eid}</text>')
            svg.append(f'<text x="{x + bar_w * 0.35:.1f}" '
                       f'y="{margin_t + plot_h + 30}" text-anchor="middle" '
                       f'font-size="9" fill="#666">(t_e={te})</text>')

        svg.append("</svg></div>")
        parts.append("\n".join(svg))

    # =================================================================
    # Section 5: Global Fisher Information profile
    # =================================================================
    parts.append("<h2>5. Global Fisher Information Profile</h2>")

    chart_w, chart_h = 500, 200
    margin_l, margin_b, margin_t = 70, 40, 30
    plot_w = chart_w - margin_l - 20
    plot_h = chart_h - margin_b - margin_t
    max_val = fisher_global.max() if fisher_global.max() > 0 else 1.0

    svg = ['<div class="card">']
    svg.append(f'<svg width="{chart_w}" height="{chart_h}" style="overflow:visible">')

    for i in range(5):
        y = margin_t + plot_h - (plot_h * i / 4)
        v = max_val * i / 4
        svg.append(f'<line x1="{margin_l}" y1="{y}" x2="{margin_l + plot_w}" '
                   f'y2="{y}" stroke="#e0e0e0" stroke-width="0.5"/>')
        svg.append(f'<text x="{margin_l - 6}" y="{y + 4}" text-anchor="end" '
                   f'font-size="10" fill="#666">{v:.1e}</text>')

    bar_w = plot_w / (T + 1)
    for ti in range(T):
        val = fisher_global[ti]
        h = (val / max_val) * plot_h
        x = margin_l + (ti + 0.5) * bar_w
        y = margin_t + plot_h - h
        ic_g = compute_ic(fisher_global)
        color = _color_for_ic(ti + 1, T)  # color by timestep position
        svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.7:.1f}" '
                   f'height="{h:.1f}" fill="{color}" rx="2"/>')
        frac = val / fisher_global.sum() * 100 if fisher_global.sum() > 0 else 0
        svg.append(f'<text x="{x + bar_w * 0.35:.1f}" y="{y - 4:.1f}" '
                   f'text-anchor="middle" font-size="9">{frac:.1f}%</text>')
        svg.append(f'<text x="{x + bar_w * 0.35:.1f}" '
                   f'y="{margin_t + plot_h + 16}" text-anchor="middle" '
                   f'font-size="11">t={ti + 1}</text>')

    ic_g = compute_ic(fisher_global)
    svg.append(f'<text x="{margin_l + plot_w / 2}" y="{margin_t - 8}" '
               f'text-anchor="middle" font-size="12" font-weight="bold">'
               f'IC = {ic_g:.3f}</text>')

    svg.append("</svg></div>")
    parts.append("\n".join(svg))

    # =================================================================
    # Section 6: Raw data table
    # =================================================================
    parts.append("<h2>6. Raw Fisher Information Values</h2>")
    parts.append('<div class="card"><table>')
    hdr = "<tr><th>Component</th>"
    for t in range(T):
        hdr += f"<th>I(t={t+1})</th>"
    hdr += "<th>IC</th></tr>"
    parts.append(hdr)

    for layer in layers:
        experts_in_layer = sorted(e for (l, e) in fisher_expert if l == layer)
        for eid in experts_in_layer:
            fi = fisher_expert[(layer, eid)]
            ic = compute_ic(fi)
            te = expert_timesteps[eid] if expert_timesteps else T
            row = f"<tr><td>L{layer} E{eid} (t_e={te})</td>"
            for t_idx, v in enumerate(fi):
                if t_idx < te:
                    row += f"<td>{v:.4e}</td>"
                else:
                    row += '<td style="color:#bbb">--</td>'
            row += f'<td style="font-weight:bold">{ic:.3f}</td></tr>'
            parts.append(row)
        if layer in fisher_gate:
            fi = fisher_gate[layer]
            ic = compute_ic(fi)
            row = f"<tr><td>L{layer} Gate</td>"
            for v in fi:
                row += f"<td>{v:.4e}</td>"
            row += f"<td>{ic:.3f}</td></tr>"
            parts.append(row)
        if layer in fisher_attn:
            fi = fisher_attn[layer]
            ic = compute_ic(fi)
            row = f"<tr><td>L{layer} Attention</td>"
            for v in fi:
                row += f"<td>{v:.4e}</td>"
            row += f"<td>{ic:.3f}</td></tr>"
            parts.append(row)

    fi = fisher_global
    ic = compute_ic(fi)
    row = "<tr style='font-weight:bold'><td>GLOBAL</td>"
    for v in fi:
        row += f"<td>{v:.4e}</td>"
    row += f"<td>{ic:.3f}</td></tr>"
    parts.append(row)

    parts.append("</table></div>")
    parts.append("</body></html>")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "tic_analysis.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/4] Loading model ...")
    mdl = load_model_and_checkpoint(args)
    expert_timesteps = get_expert_timesteps(mdl)
    T = args.time_steps
    print(f"       T={T}, expert_timesteps={expert_timesteps}")

    print("[2/4] Preparing data loader ...")
    loader = make_loader(args)

    print(f"[3/4] Computing Fisher Information over {args.num_images} images ...")
    fisher_expert, fisher_gate, fisher_attn, fisher_global, n_samples = \
        compute_fisher_info(mdl, loader, args)

    print(f"[4/4] Generating output (N={n_samples}) ...")
    print_results(fisher_expert, fisher_gate, fisher_attn, fisher_global,
                  T, expert_timesteps, n_samples)
    save_csvs(fisher_expert, fisher_gate, fisher_attn, fisher_global,
              T, expert_timesteps, args.output_dir)
    generate_html(fisher_expert, fisher_gate, fisher_attn, fisher_global,
                  T, expert_timesteps, args.output_dir, n_samples)

    print("\nDone.")


if __name__ == "__main__":
    main()


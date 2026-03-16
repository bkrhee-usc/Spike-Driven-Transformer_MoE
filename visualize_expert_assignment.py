"""Visualize hard expert assignment (token -> expert) as a grid PNG.

For a small number of input images (default: 4), saves a single PNG where:
  - rows   = images
  - col 0  = original image only
  - col 1.. = original image + layer 0, 1, ... expert assignment overlaid (semi-transparent)

Legend shows expert id and per-expert timestep (e.g. E0 (T=4), E1 (T=1)).
Uses forward hooks on each Top2Gating module (block.X.mlp.gate) and reads
its cached routing outputs (last_indices).

Example:
  python visualize_expert_assignment.py \
    -c conf/cifar100/4_384_300E_t4.yml \
    --resume /path/to/model_best.pth.tar \
    --num-images 4 \
    --output-dir ./expert_assign_vis
"""

import argparse
import os
import re
from collections import OrderedDict

import yaml
import numpy as np
import torch

from spikingjelly.clock_driven import functional
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model
from timm.models.helpers import clean_state_dict

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
    p.add_argument("--batch-size", "-b", type=int, default=1)
    p.add_argument("--workers", "-j", type=int, default=2)

    p.add_argument("--resume", required=True, type=str, help="checkpoint path")
    p.add_argument(
        "--num-images",
        type=int,
        default=4,
        help="number of images to visualize (default: 4)",
    )
    p.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="start index in the validation loader (default: 0)",
    )
    p.add_argument("--output-dir", default="./expert_assign_vis", type=str)
    p.add_argument("--device", default="cuda:0", type=str)
    p.add_argument(
        "--no-legend",
        action="store_true",
        default=False,
        help="disable legend in the output figure",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="output figure DPI (default: 200)",
    )
    p.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="alpha for expert overlay on original image (default: 0.45)",
    )

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
        num_workers=args.workers,
        distributed=False,
        crop_pct=data_config["crop_pct"],
        pin_memory=False,
    )
    return loader


def get_class_names(args):
    """Return list of human-readable class names for the dataset, if available."""
    dataset_eval = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.val_split,
        is_training=False,
        batch_size=1,
    )
    # Common attributes used by torchvision / timm datasets
    if hasattr(dataset_eval, "classes"):
        return list(dataset_eval.classes)
    if hasattr(dataset_eval, "classnames"):
        return list(dataset_eval.classnames)
    if hasattr(dataset_eval, "class_to_idx"):
        # Invert mapping and sort by index
        idx_to_class = {v: k for k, v in dataset_eval.class_to_idx.items()}
        return [idx_to_class[i] for i in range(len(idx_to_class))]
    return None


class ExpertAssignmentCapture:
    """Capture per-block hard assignment (token -> expert id).

    For each block b, stores:
      self.assignments[b] = (H, W, indices_hw)
        where indices_hw is np.int64 array of shape (H, W)
    """

    def __init__(self, net):
        self.assignments = OrderedDict()
        self.num_experts = None
        self._handles = []
        self._register(net)

    def _register(self, net):
        gate_pat = re.compile(r"(^|.*\.)block\.(\d+)\.mlp\.gate$")
        for name, module in net.named_modules():
            m = gate_pat.match(name)
            if m is None:
                continue
            block_id = int(m.group(2))
            self._handles.append(
                module.register_forward_hook(self._make_gate_hook(block_id))
            )

    def _make_gate_hook(self, block_id: int):
        def hook_fn(module, input, output):
            # module is Top2Gating. We rely on its cached routing results.
            # last_indices: list[top_k] of (B, N) tensors (top_k is 1 by default).
            if getattr(module, "last_indices", None) is None or not module.last_indices:
                return

            x_in = input[0]  # (T, B, D, H, W)
            H = int(x_in.shape[-2])
            W = int(x_in.shape[-1])

            idx = module.last_indices[0]  # (B, N)
            if idx.ndim != 2:
                return

            idx0 = idx[0].detach().cpu().to(torch.int64).numpy()
            if idx0.size != H * W:
                # If shapes mismatch (shouldn't), skip to avoid misleading plots.
                return

            idx_hw = idx0.reshape(H, W)

            # Debug: print expert id for top-left token (row=0, col=0) of
            # the first image, first layer (block 0).
            if block_id == 0:
                top_left_expert = int(idx_hw[0, 0])
                print(
                    f"[debug] block 0, token (r=0, c=0) expert id = {top_left_expert}"
                )

            self.assignments[block_id] = (H, W, idx_hw)

            if self.num_experts is None and hasattr(module, "num_gates"):
                self.num_experts = int(module.num_gates)

        return hook_fn

    def clear(self):
        self.assignments.clear()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _build_expert_cmap(num_experts: int):
    # Use a stable, high-contrast palette (tab10/tab20-like) and cycle if needed.
    base = np.array(
        [
            [228, 26, 28],
            [55, 126, 184],
            [77, 175, 74],
            [152, 78, 163],
            [255, 127, 0],
            [255, 255, 51],
            [166, 86, 40],
            [247, 129, 191],
            [153, 153, 153],
            [102, 194, 165],
            [252, 141, 98],
            [141, 160, 203],
            [231, 138, 195],
            [229, 196, 148],
            [179, 179, 179],
            [255, 217, 47],
            [102, 166, 30],
            [230, 171, 2],
            [166, 118, 29],
            [102, 102, 102],
        ],
        dtype=np.float32,
    )
    colors = np.vstack([base for _ in range((num_experts + len(base) - 1) // len(base))])[
        :num_experts
    ]
    colors = (colors / 255.0).tolist()
    from matplotlib.colors import ListedColormap

    return ListedColormap(colors, name="experts")


def _denormalize_image(img_tensor, mean, std):
    """(1,C,H,W) or (C,H,W) -> (H,W,C) uint8 in [0,255]."""
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    img = img_tensor.cpu().float()
    for c in range(img.shape[0]):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1).mul(255).byte()
    return img.permute(1, 2, 0).numpy()


def _assignment_to_rgba_overlay(idx_hw, cmap, num_experts, alpha=0.45):
    """(H,W) int indices -> (H,W,4) RGBA for overlay (values in [0,1])."""
    norm = (idx_hw.astype(np.float32) + 0.5) / max(num_experts, 1)
    rgba = cmap(norm)
    if rgba.ndim == 2:
        rgba = rgba[:, :, np.newaxis]
    if rgba.shape[-1] == 3:
        rgba = np.concatenate([rgba, np.full((*idx_hw.shape, 1), alpha)], axis=-1)
    else:
        rgba = rgba.copy()
        rgba[:, :, 3] = alpha
    return rgba


def _resize_assignment_to_image(idx_hw, target_h, target_w):
    """Upsample (H,W) to (target_h, target_w) via nearest neighbor repeat."""
    h, w = idx_hw.shape
    if h == target_h and w == target_w:
        return idx_hw
    if target_h % h != 0 or target_w % w != 0:
        # Fallback: simple repeat by integer factors then crop/pad to exact size
        sh, sw = max(1, target_h // h), max(1, target_w // w)
        out = np.repeat(np.repeat(idx_hw, sh, axis=0), sw, axis=1)
        if out.shape[0] > target_h or out.shape[1] > target_w:
            out = out[:target_h, :target_w]
        elif out.shape[0] < target_h or out.shape[1] < target_w:
            out = np.pad(
                out,
                ((0, target_h - out.shape[0]), (0, target_w - out.shape[1])),
                mode="edge",
            )
        return out
    sh, sw = target_h // h, target_w // w
    out = np.repeat(np.repeat(idx_hw, sh, axis=0), sw, axis=1)
    return out


def save_grid_png(
    grid_maps,
    num_experts: int,
    output_path: str,
    dpi: int = 200,
    show_legend: bool = True,
    original_images=None,
    expert_timesteps=None,
    overlay_alpha: float = 0.45,
    row_pred_labels=None,
    row_gt_labels=None,
    row_confidences=None,
):
    """grid_maps: list[list[np.ndarray|None]] per row: [layer0_map, layer1_map, ...].
    original_images: list of (H_img, W_img, C) uint8; if provided, col0 = image only,
        col1.. = image + layer-wise expert overlay.
    expert_timesteps: list of int (one per expert); if provided, legend shows E{e} (T=...).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    R = len(grid_maps)
    num_layers = max(len(r) for r in grid_maps) if R > 0 else 0
    has_originals = (
        original_images is not None
        and len(original_images) == R
        and all(x is not None for x in original_images)
    )
    C = (1 if has_originals else 0) + num_layers
    if R == 0 or C == 0:
        raise RuntimeError("No maps to plot.")

    cmap = _build_expert_cmap(num_experts)
    fig_w = max(6.0, 1.6 * C)
    fig_h = max(4.0, 1.6 * R)
    fig, axes = plt.subplots(R, C, figsize=(fig_w, fig_h), squeeze=False)

    for i in range(R):
        row_maps = grid_maps[i]
        img_rgb = original_images[i] if has_originals else None
        H_img = img_rgb.shape[0] if img_rgb is not None else None
        W_img = img_rgb.shape[1] if img_rgb is not None else None

        # Column 0: original image only (no overlay)
        if has_originals:
            ax = axes[i, 0]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(img_rgb)
            if i == 0:
                ax.set_title("original", fontsize=10)
            ax.set_ylabel(f"img {i}", fontsize=10)

        # Columns 1..: original image + layer-wise expert overlay
        for j in range(num_layers):
            col = (1 if has_originals else 0) + j
            ax = axes[i, col]
            ax.set_xticks([])
            ax.set_yticks([])
            m = row_maps[j] if j < len(row_maps) else None
            if has_originals and img_rgb is not None:
                ax.imshow(img_rgb)
                if m is not None and H_img is not None and W_img is not None:
                    idx_big = _resize_assignment_to_image(m, H_img, W_img)
                    overlay = _assignment_to_rgba_overlay(
                        idx_big, cmap, num_experts, alpha=overlay_alpha
                    )
                    ax.imshow(overlay, interpolation="nearest")
            else:
                if m is None:
                    ax.set_facecolor("#f2f2f2")
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=9)
                else:
                    ax.imshow(
                        m,
                        cmap=cmap,
                        vmin=-0.5,
                        vmax=num_experts - 0.5,
                        interpolation="nearest",
                    )
            if i == 0:
                ax.set_title(f"layer {j}", fontsize=10)
            if not has_originals and col == 0:
                ax.set_ylabel(f"img {i}", fontsize=10)

        # Row-level text annotation (ground-truth, prediction, confidence) on the last column
        if (
            row_pred_labels is not None
            and row_gt_labels is not None
            and row_confidences is not None
        ):
            if i < len(row_pred_labels) and i < len(row_gt_labels) and i < len(
                row_confidences
            ):
                ax_last = axes[i, C - 1]
                ax_last.text(
                    1.02,
                    0.55,
                    f"gt={row_gt_labels[i]}",
                    transform=ax_last.transAxes,
                    va="center",
                    ha="left",
                    fontsize=9,
                    color="black",
                )
                ax_last.text(
                    1.02,
                    0.35,
                    f"pred={row_pred_labels[i]} ({row_confidences[i]:.2f})",
                    transform=ax_last.transAxes,
                    va="center",
                    ha="left",
                    fontsize=9,
                    color="black",
                )

    if show_legend:
        if expert_timesteps is not None and len(expert_timesteps) >= num_experts:
            labels = [f"E{e} (T={expert_timesteps[e]})" for e in range(num_experts)]
        else:
            labels = [f"E{e}" for e in range(num_experts)]
        patches = [
            mpatches.Patch(color=cmap(e), label=labels[e]) for e in range(num_experts)
        ]
        fig.legend(
            handles=patches,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            borderaxespad=0.0,
            fontsize=9,
            title="Experts",
            title_fontsize=10,
        )
        fig.tight_layout(rect=[0, 0, 0.90, 1])
    else:
        fig.tight_layout()

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.resume} ...")
    net = load_model_and_checkpoint(args)
    loader = make_loader(args)
    class_names = get_class_names(args)
    data_config = resolve_data_config(vars(args), model=None)
    img_mean = list(data_config["mean"])
    img_std = list(data_config["std"])

    capture = ExpertAssignmentCapture(net)
    if not capture._handles:
        print("[ERROR] No gate modules found (block.X.mlp.gate).")
        for name, _ in net.named_modules():
            if "mlp.gate" in name:
                print(f"  {name}")
        return

    num_layers = int(args.layer)
    num_images = int(args.num_images)
    start_idx = int(args.start_idx)

    grid = []
    original_images = []
    row_pred_labels = []
    row_gt_labels = []
    row_confidences = []
    found_num_experts = None
    expert_timesteps = None
    for _, module in net.named_modules():
        if hasattr(module, "expert_timesteps"):
            expert_timesteps = list(module.expert_timesteps)
            break

    print(f"Running {num_images} image(s) starting at idx={start_idx} ...")
    with torch.no_grad():
        seen = 0
        for i, (img, target) in enumerate(loader):
            if i < start_idx:
                continue
            if seen >= num_images:
                break

            img = img.float().to(args.device)
            orig_np = _denormalize_image(img, img_mean, img_std)
            original_images.append(orig_np)

            functional.reset_net(net)
            capture.clear()

            output, _ = net(img, hook=dict())
            # Prediction is over class dimension; output has already been
            # averaged over timesteps inside the model (for non-TET mode).
            pred = output.argmax(dim=-1).item()
            gt = target.item()
            probs = output.softmax(dim=-1)
            conf = probs.max(dim=-1).values.item()
            if class_names is not None and pred < len(class_names) and gt < len(
                class_names
            ):
                pred_label = class_names[pred]
                gt_label = class_names[gt]
            else:
                pred_label = str(pred)
                gt_label = str(gt)
            row_pred_labels.append(pred_label)
            row_gt_labels.append(gt_label)
            row_confidences.append(conf)

            if capture.num_experts is not None:
                found_num_experts = capture.num_experts

            row = []
            for layer_id in range(num_layers):
                if layer_id in capture.assignments:
                    _, _, idx_hw = capture.assignments[layer_id]
                    row.append(idx_hw)
                else:
                    row.append(None)
            grid.append(row)
            seen += 1

    capture.remove_hooks()

    if found_num_experts is None:
        for _, module in net.named_modules():
            if hasattr(module, "num_gates"):
                found_num_experts = int(module.num_gates)
                break

    if found_num_experts is None:
        raise RuntimeError("Could not determine num_experts from gate modules.")

    out_path = os.path.join(args.output_dir, "expert_assignment_grid.png")
    save_grid_png(
        grid,
        num_experts=found_num_experts,
        output_path=out_path,
        dpi=args.dpi,
        show_legend=not args.no_legend,
        original_images=original_images,
        expert_timesteps=expert_timesteps,
        overlay_alpha=args.overlay_alpha,
        row_pred_labels=row_pred_labels,
        row_gt_labels=row_gt_labels,
        row_confidences=row_confidences,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


import argparse
import time
import yaml
import os
import logging
import numpy as np
import random as rd
import hashlib
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from typing import List

from spikingjelly.clock_driven import functional
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import torchinfo

from timm.data import (
    create_dataset,
    create_loader,
    resolve_data_config,
    Mixup,
    FastCollateMixup,
    AugMixDataset,
)
from timm.models import (
    create_model,
    safe_model_name,
    load_checkpoint,
    model_parameters,
)
from timm.models.helpers import clean_state_dict
from timm.utils import *
from timm.loss import (
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
    JsdCrossEntropy,
    BinaryCrossEntropy,
)
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

import model, dvs_utils, criterion

try:
    # `convert_splitbn_model` was removed in newer timm versions; provide a no-op fallback.
    from timm.models import convert_splitbn_model  # type: ignore[attr-defined]
except Exception:

    def convert_splitbn_model(m, *args, **kwargs):
        return m


try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False


def resume_checkpoint(
    model, checkpoint_path, optimizer=None, loss_scaler=None, log_info=True
):
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            if log_info:
                _logger.info("Restoring model state from checkpoint...")
            state_dict = clean_state_dict(checkpoint["state_dict"])
            model.load_state_dict(state_dict, strict=False)

            if optimizer is not None and "optimizer" in checkpoint:
                if log_info:
                    _logger.info("Restoring optimizer state from checkpoint...")
                optimizer.load_state_dict(checkpoint["optimizer"])

            if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
                if log_info:
                    _logger.info("Restoring AMP loss scaler state from checkpoint...")
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

            if "epoch" in checkpoint:
                resume_epoch = checkpoint["epoch"]
                if "version" in checkpoint and checkpoint["version"] > 1:
                    resume_epoch += 1

            if log_info:
                _logger.info(
                    "Loaded checkpoint '{}' (epoch {})".format(
                        checkpoint_path, checkpoint["epoch"]
                    )
                )
        else:
            model.load_state_dict(checkpoint)
            if log_info:
                _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return resume_epoch
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_pretrain_model_only(model: nn.Module, checkpoint_path: str, log_info: bool = True):
    """Load timm-style checkpoint weights into model only (no optimizer/scheduler/scaler)."""
    if not checkpoint_path:
        return
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not (isinstance(checkpoint, dict) and "state_dict" in checkpoint):
        raise ValueError("Expected timm-style checkpoint dict with key 'state_dict'.")
    if log_info:
        _logger.info(f"Loading pretrain weights (model-only) from '{checkpoint_path}'")
    state_dict = clean_state_dict(checkpoint["state_dict"])
    model.load_state_dict(state_dict, strict=False)


def _ckpt_hash_tag(path: str, n: int = 8) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


torch.backends.cudnn.benchmark = True

config_parser = parser = argparse.ArgumentParser(
    description="Finetune Config", add_help=False
)
parser.add_argument(
    "-c",
    "--config",
    default="imagenet.yml",
    type=str,
    metavar="FILE",
    help="YAML config file specifying default arguments",
)

parser = argparse.ArgumentParser(description="Finetune w/ Router Self-KD")

# Dataset / Model parameters
parser.add_argument(
    "-data-dir",
    metavar="DIR",
    default="/scratch1/bkrhee/data",
    help="path to dataset",
)
parser.add_argument(
    "--dataset",
    "-d",
    metavar="NAME",
    default="torch/cifar10",
    help="dataset type (default: ImageFolder/ImageTar if empty)",
)
parser.add_argument(
    "--train-split",
    metavar="NAME",
    default="train",
    help="dataset train split (default: train)",
)
parser.add_argument(
    "--val-split",
    metavar="NAME",
    default="validation",
    help="dataset validation split (default: validation)",
)
parser.add_argument(
    "--train-split-path",
    type=str,
    default=None,
    metavar="N",
    help="",
)
parser.add_argument(
    "--model",
    default="sdt",
    type=str,
    metavar="MODEL",
    help='Name of model to finetune (default: "sdt")',
)
parser.add_argument(
    "--pooling-stat",
    default="1111",
    type=str,
    help="pooling layers in SPS moduls",
)
parser.add_argument("--TET", default=False, type=bool, help="")
parser.add_argument("--TET-means", default=1.0, type=float, help="")
parser.add_argument("--TET-lamb", default=0.0, type=float, help="")
parser.add_argument("--spike-mode", default="lif", type=str, help="")
parser.add_argument("--layer", default=4, type=int, help="")
parser.add_argument("--in-channels", default=3, type=int, help="")

parser.add_argument(
    "--finetune",
    default="",
    type=str,
    metavar="PATH",
    help="Load model weights from timm-style checkpoint and start finetuning from epoch 0 (model-only).",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="Resume finetuning from finetune checkpoint (model+optimizer+scaler state).",
)
parser.add_argument(
    "--no-resume-opt",
    action="store_true",
    default=False,
    help="prevent resume of optimizer state when resuming model",
)

parser.add_argument(
    "--pretrained",
    action="store_true",
    default=False,
    help="Start with pretrained version of specified network (if avail)",
)

parser.add_argument("--num-classes", type=int, default=1000, metavar="N", help="")
parser.add_argument("--time-steps", type=int, default=4, metavar="N", help="")
parser.add_argument("--num-heads", type=int, default=8, metavar="N", help="")
parser.add_argument("--patch-size", type=int, default=None, metavar="N", help="Image patch size")
parser.add_argument(
    "--mlp-ratio",
    type=float,
    default=4.0,
    metavar="N",
    help="expand ratio of embedding dimension in MLP block (per-expert in MoE)",
)
parser.add_argument("--num-experts", type=int, default=4, metavar="N", help="")
parser.add_argument(
    "--expert-timesteps",
    default=None,
    help="MoE expert timesteps per expert (from config: list of int).",
)
parser.add_argument("--gp", default=None, type=str, metavar="POOL", help="")
parser.add_argument("--img-size", type=int, default=None, metavar="N", help="")
parser.add_argument("--input-size", default=None, nargs=3, type=int, metavar="N N N", help="")
parser.add_argument("--crop-pct", default=None, type=float, metavar="N", help="")
parser.add_argument("--mean", type=float, nargs="+", default=None, metavar="MEAN", help="")
parser.add_argument("--std", type=float, nargs="+", default=None, metavar="STD", help="")
parser.add_argument("--interpolation", default="", type=str, metavar="NAME", help="")
parser.add_argument("-b", "--batch-size", type=int, default=32, metavar="N", help="")
parser.add_argument("-vb", "--val-batch-size", type=int, default=16, metavar="N", help="")

# Optimizer parameters
parser.add_argument("--opt", default="sgd", type=str, metavar="OPTIMIZER", help="")
parser.add_argument("--opt-eps", default=None, type=float, metavar="EPSILON", help="")
parser.add_argument("--opt-betas", default=None, type=float, nargs="+", metavar="BETA", help="")
parser.add_argument("--momentum", type=float, default=0.9, metavar="M", help="")
parser.add_argument("--weight-decay", type=float, default=0.0001, help="")
parser.add_argument("--clip-grad", type=float, default=None, metavar="NORM", help="")
parser.add_argument("--clip-mode", type=str, default="norm", help="")

# Learning rate schedule parameters
parser.add_argument("--sched", default="step", type=str, metavar="SCHEDULER", help="")
parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="")
parser.add_argument("--lr-noise", type=float, nargs="+", default=None, metavar="pct, pct", help="")
parser.add_argument("--lr-noise-pct", type=float, default=0.67, metavar="PERCENT", help="")
parser.add_argument("--lr-noise-std", type=float, default=1.0, metavar="STDDEV", help="")
parser.add_argument("--lr-cycle-mul", type=float, default=1.0, metavar="MULT", help="")
parser.add_argument("--lr-cycle-limit", type=int, default=1, metavar="N", help="")
parser.add_argument("--warmup-lr", type=float, default=0.0001, metavar="LR", help="")
parser.add_argument("--min-lr", type=float, default=1e-5, metavar="LR", help="")
parser.add_argument("--epochs", type=int, default=50, metavar="N", help="finetune epochs")
parser.add_argument("--epoch-repeats", type=float, default=0.0, metavar="N", help="")
parser.add_argument("--start-epoch", default=None, type=int, metavar="N", help="(resume only)")
parser.add_argument("--decay-epochs", type=float, default=30, metavar="N", help="")
parser.add_argument("--warmup-epochs", type=int, default=3, metavar="N", help="")
parser.add_argument("--cooldown-epochs", type=int, default=10, metavar="N", help="")
parser.add_argument("--patience-epochs", type=int, default=10, metavar="N", help="")
parser.add_argument("--decay-rate", "--dr", type=float, default=0.1, metavar="RATE", help="")

# Augmentation & regularization parameters
parser.add_argument("--no-aug", action="store_true", default=False, help="")
parser.add_argument("--scale", type=float, nargs="+", default=[0.08, 1.0], metavar="PCT", help="")
parser.add_argument("--ratio", type=float, nargs="+", default=[3.0 / 4.0, 4.0 / 3.0], metavar="RATIO", help="")
parser.add_argument("--hflip", type=float, default=0.5, help="")
parser.add_argument("--vflip", type=float, default=0.0, help="")
parser.add_argument("--color-jitter", type=float, default=0.4, metavar="PCT", help="")
parser.add_argument("--aa", type=str, default=None, metavar="NAME", help="")
parser.add_argument("--aug-splits", type=int, default=0, help="")
parser.add_argument("--jsd", action="store_true", default=False, help="")
parser.add_argument("--bce-loss", action="store_true", default=False, help="")
parser.add_argument("--bce-target-thresh", type=float, default=None, help="")
parser.add_argument("--reprob", type=float, default=0.0, metavar="PCT", help="")
parser.add_argument("--remode", type=str, default="const", help="")
parser.add_argument("--recount", type=int, default=1, help="")
parser.add_argument("--resplit", action="store_true", default=False, help="")
parser.add_argument("--mixup", type=float, default=0.0, help="")
parser.add_argument("--cutmix", type=float, default=0.0, help="")
parser.add_argument("--cutmix-minmax", type=float, nargs="+", default=None, help="")
parser.add_argument("--mixup-prob", type=float, default=1.0, help="")
parser.add_argument("--mixup-switch-prob", type=float, default=0.5, help="")
parser.add_argument("--mixup-mode", type=str, default="batch", help="")
parser.add_argument("--mixup-off-epoch", default=0, type=int, metavar="N", help="")
parser.add_argument("--smoothing", type=float, default=0.1, help="")
parser.add_argument("--train-interpolation", type=str, default="random", help="")
parser.add_argument("--drop", type=float, default=0.0, metavar="PCT", help="")
parser.add_argument("--drop-connect", type=float, default=None, metavar="PCT", help="")
parser.add_argument("--drop-path", type=float, default=0.2, metavar="PCT", help="")
parser.add_argument("--drop-block", type=float, default=None, metavar="PCT", help="")

# Batch norm / DDP misc
parser.add_argument("--bn-tf", action="store_true", default=False, help="")
parser.add_argument("--bn-momentum", type=float, default=None, help="")
parser.add_argument("--bn-eps", type=float, default=None, help="")
parser.add_argument("--sync-bn", action="store_true", help="")
parser.add_argument("--dist-bn", type=str, default="", help="")
parser.add_argument("--split-bn", action="store_true", help="")
parser.add_argument("--linear-prob", action="store_true", help="")

# EMA
parser.add_argument("--model-ema", action="store_true", default=False, help="")
parser.add_argument("--model-ema-force-cpu", action="store_true", default=False, help="")
parser.add_argument("--model-ema-decay", type=float, default=0.9998, help="")

# Misc
parser.add_argument("--seed", type=int, default=42, metavar="S", help="")
parser.add_argument("--log-interval", type=int, default=100, metavar="N", help="")
parser.add_argument("--recovery-interval", type=int, default=0, metavar="N", help="")
parser.add_argument("--checkpoint-hist", type=int, default=10, metavar="N", help="")
parser.add_argument("-j", "--workers", type=int, default=4, metavar="N", help="")
parser.add_argument("--save-images", action="store_true", default=False, help="")
parser.add_argument("--amp", action="store_true", default=False, help="")
parser.add_argument("--apex-amp", action="store_true", default=False, help="")
parser.add_argument("--native-amp", action="store_true", default=False, help="")
parser.add_argument("--channels-last", action="store_true", default=False, help="")
parser.add_argument("--pin-mem", action="store_true", default=False, help="")
parser.add_argument("--no-prefetcher", action="store_true", default=False, help="")
parser.add_argument("--dvs-aug", action="store_true", default=False, help="")
parser.add_argument("--dvs-trival-aug", action="store_true", default=False, help="")
parser.add_argument(
    "--output",
    default="./output/finetune",
    type=str,
    metavar="PATH",
    help="finetune output base dir",
)
parser.add_argument("--experiment", default="", type=str, metavar="NAME", help="")
parser.add_argument("--eval-metric", default="top1", type=str, metavar="EVAL_METRIC", help="")
parser.add_argument("--tta", type=int, default=0, metavar="N", help="")
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--use-multi-epochs-loader", action="store_true", default=False, help="")
parser.add_argument("--torchscript", dest="torchscript", action="store_true", help="")
parser.add_argument("--log-wandb", action="store_true", default=False, help="")

# Router self-KD (same as train_routerKD.py)
parser.add_argument(
    "--router-kd",
    action="store_true",
    default=False,
    help="Enable self knowledge distillation on MoE routers: blocks[0..L-2] match blocks[L-1] routing.",
)
parser.add_argument("--router-kd-weight", type=float, default=1.0, help="Base weight for router KD loss.")
parser.add_argument(
    "--router-kd-temp",
    type=float,
    default=2.0,
    help="Temperature for router KD; applied to probability outputs via softmax(log(p)/T).",
)
parser.add_argument(
    "--router-kd-warmup",
    type=str,
    default="none",
    choices=["none", "linear"],
    help="Warmup schedule for router KD weight.",
)
parser.add_argument("--router-kd-warmup-epochs", type=int, default=0, help="Warmup epochs for router KD.")


_logger = logging.getLogger("finetune")
stream_handler = logging.StreamHandler()
format_str = "%(asctime)s %(levelname)s: %(message)s"
stream_handler.setFormatter(logging.Formatter(format_str))
_logger.addHandler(stream_handler)
_logger.propagate = False


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    # Avoid trying to open default config on `--help`
    if args_config.config and ("-h" not in remaining and "--help" not in remaining):
        if os.path.isfile(args_config.config):
            with open(args_config.config, "r") as f:
                cfg = yaml.safe_load(f)
                parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def _unwrap_model(m: nn.Module) -> nn.Module:
    return getattr(m, "module", m)


def _router_kd_weight_for_epoch(epoch: int, args) -> float:
    base = float(getattr(args, "router_kd_weight", 1.0))
    if not getattr(args, "router_kd", False):
        return 0.0
    warmup = str(getattr(args, "router_kd_warmup", "none")).lower()
    warmup_epochs = int(getattr(args, "router_kd_warmup_epochs", 0) or 0)
    if warmup == "linear" and warmup_epochs > 0:
        scale = min(1.0, max(0.0, (epoch + 1) / float(warmup_epochs)))
        return base * scale
    return base


def _apply_temp_to_probs(probs: torch.Tensor, temp: float, eps: float = 1e-8) -> torch.Tensor:
    if temp is None or float(temp) == 1.0:
        return probs.clamp_min(eps)
    logits = (probs.clamp_min(eps)).log() / float(temp)
    return logits.softmax(dim=-1)


def router_self_kd_kl(
    teacher_probs: torch.Tensor,
    student_probs_list: List[torch.Tensor],
    temp: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    if teacher_probs is None or not student_probs_list:
        return torch.zeros((), device=teacher_probs.device if teacher_probs is not None else "cuda")
    p_t = _apply_temp_to_probs(teacher_probs.detach(), temp=temp, eps=eps)
    log_p_t = (p_t.clamp_min(eps)).log()
    kd_losses = []
    for p_s_raw in student_probs_list:
        if p_s_raw is None:
            continue
        p_s = _apply_temp_to_probs(p_s_raw, temp=temp, eps=eps)
        log_p_s = (p_s.clamp_min(eps)).log()
        kl = (p_t * (log_p_t - log_p_s)).sum(dim=-1)
        kd_losses.append(kl.mean())
    if not kd_losses:
        return torch.zeros((), device=teacher_probs.device)
    kd = torch.stack(kd_losses).mean()
    return kd * (float(temp) ** 2)


def main():
    setup_default_logging()
    args, args_text = _parse_args()

    if args.finetune and args.resume:
        raise ValueError("Use either --finetune (start from pretrain weights) OR --resume (resume finetune), not both.")
    if not args.finetune and not args.resume:
        raise ValueError("Specify --finetune <pretrain_ckpt> to start or --resume <finetune_ckpt> to continue.")

    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning("wandb logging requested but wandb not installed.")

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.device = "cuda:0"
    args.world_size = 1
    args.rank = 0
    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    assert args.rank >= 0

    # resolve AMP args
    use_amp = None
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = "apex"
    elif args.native_amp and has_native_amp:
        use_amp = "native"

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.initial_seed()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random_seed(args.seed, args.rank)
    torch.backends.cudnn.deterministic = True
    rd.seed(args.seed)

    args.dvs_mode = args.dataset in ["cifar10-dvs-tet", "cifar10-dvs"]

    model = create_model(
        args.model,
        T=args.time_steps,
        pretrained=args.pretrained,
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
        num_experts=args.num_experts,
        expert_timesteps=getattr(args, "expert_timesteps", None),
        in_channels=args.in_channels,
        qkv_bias=False,
        depths=args.layer,
        sr_ratios=1,
        spike_mode=args.spike_mode,
        dvs_mode=args.dvs_mode,
        TET=args.TET,
    )

    # Load pretrain model weights (model-only) before optimizer creation
    if args.finetune:
        load_pretrain_model_only(model, args.finetune, log_info=args.rank == 0)

    data_config = resolve_data_config(vars(args), model=model, verbose=args.rank == 0)

    # output dir policy:
    # - resume: reuse checkpoint directory
    # - finetune: create new dir under ./output/finetune and tag with ckpt hash
    output_dir = None
    if args.rank == 0:
        if args.resume:
            output_dir = os.path.dirname(args.resume)
            os.makedirs(output_dir, exist_ok=True)
            exp_name = os.path.basename(output_dir.rstrip("/"))
        else:
            ckpt_tag = _ckpt_hash_tag(args.finetune, n=8)
            if args.experiment:
                exp_name = f"{args.experiment}-ft-ckpt{ckpt_tag}"
            else:
                exp_name = "-".join(
                    [
                        datetime.now().strftime("%Y%m%d-%H%M%S"),
                        safe_model_name(args.model),
                        "data-" + args.dataset.split("/")[-1],
                        f"ft-ckpt{ckpt_tag}",
                        f"t-{args.time_steps}",
                        f"spike-{args.spike_mode}",
                    ]
                )
            output_dir = get_outdir(args.output if args.output else "./output/finetune", exp_name)

        file_handler = logging.FileHandler(os.path.join(output_dir, f"{args.model}.log"), "a")
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(logging.INFO)
        _logger.addHandler(file_handler)

        with open(os.path.join(output_dir, "args.yaml"), "w") as f:
            f.write(args_text)
            if args.finetune:
                f.write(f"\nfinetune_ckpt: {args.finetune}\n")
            if args.resume:
                f.write(f"\nresume_ckpt: {args.resume}\n")

    # split-bn support
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1
        num_aug_splits = args.aug_splits
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp != "native":
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.torchscript:
        assert not use_amp == "apex"
        assert not args.sync_bn
        model = torch.jit.script(model)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    amp_autocast = suppress
    loss_scaler = None
    if use_amp == "apex":
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        loss_scaler = ApexScaler()
    elif use_amp == "native":
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    # resume finetune checkpoint (full state) if requested
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.rank == 0,
        )

    model_ema = None
    if args.model_ema:
        model_ema = ModelEmaV2(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else None,
        )
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    if args.distributed:
        if has_apex and use_amp != "native":
            model = ApexDDP(model, delay_allreduce=True, find_unused_parameters=True)
        else:
            model = NativeDDP(model, device_ids=[args.local_rank], find_unused_parameters=True)

    if args.linear_prob:
        for n, p in model.module.named_parameters():
            if "patch_embed" in n:
                p.requires_grad = False

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        start_epoch = args.start_epoch
    elif resume_epoch is not None and (not args.linear_prob):
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    transforms_train, transforms_eval = None, None
    dataset_train, dataset_eval = None, None
    if args.dataset == "cifar10-dvs-tet":
        dataset_train = dvs_utils.DVSCifar10(root=os.path.join(args.data_dir, "train"), train=True)
        dataset_eval = dvs_utils.DVSCifar10(root=os.path.join(args.data_dir, "test"), train=False)
    elif args.dataset == "cifar10-dvs":
        dataset = CIFAR10DVS(
            args.data_dir,
            data_type="frame",
            frames_number=args.time_steps,
            split_by="number",
            transform=dvs_utils.Resize(64),
        )
        dataset_train, dataset_eval = dvs_utils.split_to_train_test_set(0.9, dataset, 10)
    elif args.dataset == "gesture":
        dataset_train = DVS128Gesture(
            args.data_dir,
            train=True,
            data_type="frame",
            frames_number=args.time_steps,
            split_by="number",
        )
        dataset_eval = DVS128Gesture(
            args.data_dir,
            train=False,
            data_type="frame",
            frames_number=args.time_steps,
            split_by="number",
        )
    else:
        dataset_train = create_dataset(
            args.dataset,
            root=args.data_dir,
            split=args.train_split,
            is_training=True,
            batch_size=args.batch_size,
            repeats=args.epoch_repeats,
            transform=transforms_train,
        )
        dataset_eval = create_dataset(
            args.dataset,
            root=args.data_dir,
            split=args.val_split,
            is_training=False,
            batch_size=args.batch_size,
            transform=transforms_eval,
        )

    collate_fn = None
    train_dvs_aug, train_dvs_trival_aug = None, None
    if args.dvs_aug:
        train_dvs_aug = dvs_utils.Cutout(n_holes=1, length=16)
    if args.dvs_trival_aug:
        train_dvs_trival_aug = dvs_utils.SNNAugmentWide()

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes,
        )
        if args.prefetcher and args.dataset not in dvs_utils.DVS_DATASET:
            assert not num_aug_splits
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    if num_aug_splits > 1 and args.dataset not in dvs_utils.DVS_DATASET:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config["interpolation"]

    if args.dataset in dvs_utils.DVS_DATASET:
        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
        )
        loader_eval = torch.utils.data.DataLoader(
            dataset_eval,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
    else:
        loader_train = create_loader(
            dataset_train,
            input_size=data_config["input_size"],
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=args.prefetcher,
            no_aug=args.no_aug,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            re_split=args.resplit,
            scale=args.scale,
            ratio=args.ratio,
            hflip=args.hflip,
            vflip=args.vflip,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            num_aug_splits=num_aug_splits,
            interpolation=train_interpolation,
            mean=data_config["mean"],
            std=data_config["std"],
            num_workers=args.workers,
            distributed=args.distributed,
            collate_fn=collate_fn,
            pin_memory=args.pin_mem,
            use_multi_epochs_loader=args.use_multi_epochs_loader,
        )
        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config["input_size"],
            batch_size=args.val_batch_size,
            is_training=False,
            use_prefetcher=args.prefetcher,
            interpolation=data_config["interpolation"],
            mean=data_config["mean"],
            std=data_config["std"],
            num_workers=args.workers,
            distributed=args.distributed,
            crop_pct=data_config["crop_pct"],
            pin_memory=args.pin_mem,
        )

    if args.jsd:
        assert num_aug_splits > 1
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
    elif mixup_active:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    saver = None
    if args.rank == 0:
        saver = CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=False,
            max_history=args.checkpoint_hist,
        )

    best_metric = None
    best_epoch = None
    eval_metric = args.eval_metric
    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and hasattr(loader_train.sampler, "set_epoch"):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fn,
                args,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                model_ema=model_ema,
                mixup_fn=mixup_fn,
                dvs_aug=train_dvs_aug,
                dvs_trival_aug=train_dvs_trival_aug,
            )

            eval_metrics = validate(
                model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast
            )

            if model_ema is not None and not args.model_ema_force_cpu:
                ema_eval_metrics = validate(
                    model_ema.module,
                    loader_eval,
                    validate_loss_fn,
                    args,
                    amp_autocast=amp_autocast,
                    log_suffix=" (EMA)",
                )
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None:
                update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    os.path.join(output_dir, "summary.csv"),
                    write_header=best_metric is None,
                    log_wandb=args.log_wandb and has_wandb,
                )

            if saver is not None:
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)
                _logger.info(
                    "*** Best metric: {0} (epoch {1})".format(best_metric, best_epoch)
                )

    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        _logger.info("*** Best metric: {0} (epoch {1})".format(best_metric, best_epoch))


def train_one_epoch(
    epoch,
    model,
    loader,
    optimizer,
    loss_fn,
    args,
    lr_scheduler=None,
    saver=None,
    output_dir=None,
    amp_autocast=suppress,
    loss_scaler=None,
    model_ema=None,
    mixup_fn=None,
    dvs_aug=None,
    dvs_trival_aug=None,
):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher:
            if hasattr(loader, "mixup_enabled"):
                loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    sample_number = 0
    start_time = time.time()

    second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    kd_losses_m = AverageMeter()

    model.train()
    functional.reset_net(model)

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        input = input.float()
        if not args.prefetcher or args.dataset in dvs_utils.DVS_DATASET:
            if args.amp and not isinstance(input, torch.cuda.HalfTensor):
                input = input.half()
            input, target = input.cuda(), target.cuda()
            if dvs_aug is not None:
                input = dvs_aug(input)
            if dvs_trival_aug is not None:
                output = []
                for i in range(input.shape[0]):
                    output.append(dvs_trival_aug(input[i]))
                input = torch.stack(output)
                del output
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)

        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            hook = {}
            output, hook = model(input, hook=hook)
            if args.TET:
                loss = criterion.TET_loss(
                    output, target, loss_fn, means=args.TET_means, lamb=args.TET_lamb
                )
            else:
                loss = loss_fn(output, target)

            moe_losses = [v for k, v in hook.items() if k.startswith("moe_loss_layer_")]
            if moe_losses:
                loss = loss + torch.stack(moe_losses).sum()

            kd_loss = None
            kd_w = 0.0
            if getattr(args, "router_kd", False):
                kd_w = _router_kd_weight_for_epoch(epoch, args)
                if kd_w > 0:
                    m = _unwrap_model(model)
                    try:
                        blocks = getattr(m, "block", None)
                        if blocks is not None and len(blocks) >= 2:
                            teacher = blocks[-1].mlp.gate.last_raw_gates
                            students = [blocks[i].mlp.gate.last_raw_gates for i in range(len(blocks) - 1)]
                            kd_loss = router_self_kd_kl(
                                teacher_probs=teacher,
                                student_probs_list=students,
                                temp=float(getattr(args, "router_kd_temp", 2.0)),
                                eps=1e-8,
                            )
                            loss = loss + (kd_w * kd_loss)
                    except Exception as e:
                        if args.local_rank == 0 and batch_idx % args.log_interval == 0:
                            _logger.warning(f"Router KD skipped due to error: {e}")

        sample_number += input.shape[0]
        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))
            if kd_loss is not None:
                kd_losses_m.update(float(kd_loss.detach().item()), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                clip_grad=args.clip_grad,
                clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head="agc" in args.clip_mode),
                create_graph=second_order,
            )
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head="agc" in args.clip_mode),
                    value=args.clip_grad,
                    mode=args.clip_mode,
                )
            optimizer.step()

        functional.reset_net(model)
        if model_ema is not None:
            model_ema.update(model)
            functional.reset_net(model_ema)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group["lr"] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            if args.local_rank == 0:
                _logger.info(
                    "Train: {} [{:>4d}/{} ({:>3.0f}%)]  "
                    "Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  "
                    "KD: {kd.val:>9.6f} ({kd.avg:>6.4f})  "
                    "KD_w: {kdw:.3f}  "
                    "Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  "
                    "({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  "
                    "LR: {lr:.3e}  "
                    "Data: {data_time.val:.3f} ({data_time.avg:.3f})".format(
                        epoch,
                        batch_idx,
                        len(loader),
                        100.0 * batch_idx / last_idx,
                        loss=losses_m,
                        kd=kd_losses_m,
                        kdw=float(kd_w) if getattr(args, "router_kd", False) else 0.0,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m,
                    )
                )

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, "train-batch-%d.jpg" % batch_idx),
                        padding=0,
                        normalize=True,
                    )

        if saver is not None and args.recovery_interval and (
            last_batch or (batch_idx + 1) % args.recovery_interval == 0
        ):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()

    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()
    if args.local_rank == 0:
        _logger.info(f"samples / s = {sample_number / (time.time() - start_time): .3f}")
    metrics = OrderedDict([("loss", losses_m.avg)])
    if getattr(args, "router_kd", False):
        metrics["kd_loss"] = kd_losses_m.avg
    return metrics


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=""):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            input = input.float()
            last_batch = batch_idx == last_idx
            if not args.prefetcher or args.dataset in dvs_utils.DVS_DATASET:
                if args.amp and not isinstance(input, torch.cuda.HalfTensor):
                    input = input.half()
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if args.TET:
                output = output.mean(0)

            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0 : target.size(0) : reduce_factor]

            loss = loss_fn(output, target)
            functional.reset_net(model)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()
            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))
            batch_time_m.update(time.time() - end)
            end = time.time()

            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = "Test" + log_suffix
                _logger.info(
                    "{0}: [{1:>4d}/{2}]  "
                    "Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                    "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                    "Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  "
                    "Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})".format(
                        log_name,
                        batch_idx,
                        last_idx,
                        batch_time=batch_time_m,
                        loss=losses_m,
                        top1=top1_m,
                        top5=top5_m,
                    )
                )

    metrics = OrderedDict([("loss", losses_m.avg), ("top1", top1_m.avg), ("top5", top5_m.avg)])
    return metrics


if __name__ == "__main__":
    main()


"""
Training script for PVSNet on the Blender view-synthesis dataset.

The model learns to map (centre view, target translation) -> translated view,
using the split files ``lf_train.txt`` / ``lf_validate.txt``.

Single GPU:
    python train.py --resolution 256x256

Multi GPU (one process per GPU, DistributedDataParallel):
    torchrun --standalone --nproc_per_node=8 train.py --resolution 256x256

Resuming:
    python train.py --resume ./checkpoint/<run>/checkpoint_blender_pvsnet_256x256.pth

Model weights are written as a bare ``state_dict`` under the same filename
convention ``test.py`` expects, so evaluation and the inference apps keep
working unchanged. Optimizer/scheduler/epoch state lives in a separate
``*_training_state.pth`` file next to it.
"""

import argparse
import contextlib
import json
import logging
import os
import random
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import parameters as params
from blender_dataset import BlenderViewSynthesisDataset
from losses import CombinedLoss
from models.pvsnet_model import PVSNet
from optimizers import SOAP
from schedulers import build_scheduler

# Metrics where a larger value is better.
HIGHER_IS_BETTER = {"psnr", "ssim", "ms_ssim"}

LOGGER = logging.getLogger("pvsnet.train")


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PVSNet on the Blender dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument("--train_file", type=str, default="./lf_train.txt")
    data.add_argument("--val_file", type=str, default="./lf_validate.txt")
    data.add_argument("--data_root", type=str, default=".",
                      help="Directory the relative paths in the split files resolve against")
    data.add_argument("--resolution", type=str, default="256x256",
                      help="Training resolution as HxW, e.g. 256x256 or 512x512")
    data.add_argument("--num_workers", type=int, default=8,
                      help="Data loader workers per process")
    data.add_argument("--no_cache_source", action="store_true",
                      help="Disable the per-worker cache of decoded centre views")
    data.add_argument("--hflip_prob", type=float, default=0.0,
                      help="Horizontal flip probability (also negates the x translation)")
    data.add_argument("--max_train_samples", type=int, default=None,
                      help="Truncate the training split (smoke tests)")
    data.add_argument("--max_val_samples", type=int, default=None,
                      help="Truncate the validation split")

    model = parser.add_argument_group("model")
    model.add_argument("--architecture", type=str, default="regular",
                       choices=["regular", "lite"])
    model.add_argument("--pose_dims", type=int, default=3)
    model.add_argument("--channels_last", action="store_true",
                       help="Use channels_last memory format (faster on tensor cores)")
    model.add_argument("--compile", action="store_true", help="torch.compile the model")

    optim_group = parser.add_argument_group("optimization")
    optim_group.add_argument("--epochs", type=int, default=100)
    optim_group.add_argument("--batch_size", type=int, default=32,
                             help="Per-GPU batch size; global batch is this x GPUs x accumulation")
    optim_group.add_argument("--val_batch_size", type=int, default=None,
                             help="Defaults to --batch_size")
    optim_group.add_argument("--grad_accum_steps", type=int, default=1)
    optim_group.add_argument("--optimizer", type=str, default="soap",
                             choices=["soap", "adamw"],
                             help="adamw is a lighter fallback when SOAP's preconditioners "
                                  "do not fit on a small GPU")
    optim_group.add_argument("--lr", type=float, default=1e-4,
                             help="SOAP's own default of 3e-3 is tuned for transformers "
                                  "and diverges on this architecture, whose residual "
                                  "blocks carry almost no normalisation")
    optim_group.add_argument("--betas", type=float, nargs=2, default=(0.95, 0.95))
    optim_group.add_argument("--weight_decay", type=float, default=0.01)
    optim_group.add_argument("--eps", type=float, default=1e-8)
    optim_group.add_argument("--precondition_frequency", type=int, default=10,
                             help="SOAP: eigenbasis refresh interval in steps")
    optim_group.add_argument("--max_precond_dim", type=int, default=2048,
                             help="SOAP: dimensions above this are left unpreconditioned")
    optim_group.add_argument("--no_merge_dims", action="store_true",
                             help="SOAP: do not merge tensor dims before preconditioning")
    optim_group.add_argument("--precondition_1d", action="store_true",
                             help="SOAP: also precondition 1D parameters")
    optim_group.add_argument("--soap_min_rank_ratio", type=float, default=0.5,
                             help="SOAP: skip preconditioning for dimensions whose "
                                  "Gram matrix rank falls below this fraction of its "
                                  "size, since their eigenvectors are numerical noise; "
                                  "0 reproduces the reference implementation")
    optim_group.add_argument("--soap_update_clip", type=float, default=10.0,
                             help="SOAP: element-wise bound on the Adam ratio, which "
                                  "stops a stale second moment from producing an "
                                  "enormous update after an eigenbasis refresh; 0 "
                                  "reproduces the reference implementation exactly")
    optim_group.add_argument("--grad_clip", type=float, default=1.0,
                             help="Gradient norm clipping; 0 disables it")
    optim_group.add_argument("--skip_limit", type=int, default=25,
                             help="Abort after this many consecutive non-finite steps; "
                                  "0 keeps skipping forever")
    optim_group.add_argument("--amp", type=str, default="bf16",
                             choices=["off", "bf16", "fp16"],
                             help="Mixed precision for the forward/backward pass")

    sched = parser.add_argument_group("scheduler")
    sched.add_argument("--scheduler", type=str, default="step",
                       choices=["step", "cosine"])
    sched.add_argument("--warmup_steps", type=int, default=500)
    sched.add_argument("--step_size_epochs", type=int, default=30,
                       help="StepLR decay period in epochs")
    sched.add_argument("--gamma", type=float, default=0.5, help="StepLR decay factor")
    sched.add_argument("--min_lr_ratio", type=float, default=0.01,
                       help="Cosine floor as a fraction of the base LR")

    loss = parser.add_argument_group("loss weights")
    loss.add_argument("--w_msssim_l1", type=float, default=1.0,
                      help="Weight of the MS-SSIM + L1 mix loss")
    loss.add_argument("--w_l2", type=float, default=0.1)
    loss.add_argument("--w_ffl", type=float, default=1.0,
                      help="Weight of the focal frequency loss. Its raw value is ~100x "
                           "smaller than the other terms, so this acts as a mild "
                           "high-frequency regulariser; raise it to ~10 to push harder")
    loss.add_argument("--w_vgg", type=float, default=0.1)
    loss.add_argument("--w_dino", type=float, default=0.05)
    loss.add_argument("--msssim_alpha", type=float, default=0.84,
                      help="MS-SSIM share inside the mix loss")
    loss.add_argument("--ffl_alpha", type=float, default=1.0)
    loss.add_argument("--ffl_patch_factor", type=int, default=1)
    loss.add_argument("--vgg_resize", type=int, default=None,
                      help="Resize inputs before the VGG loss (saves memory at 512x512)")
    loss.add_argument("--dino_model", type=str, default="dinov2_vits14")
    loss.add_argument("--dino_input_size", type=int, default=224)

    ckpt = parser.add_argument_group("checkpointing")
    ckpt.add_argument("--checkpoint_dir", type=str, default="./checkpoint")
    ckpt.add_argument("--run_name", type=str, default=None,
                      help="Defaults to blender_<arch>_<res>_<timestamp>")
    ckpt.add_argument("--resume", type=str, default=None,
                      help="Checkpoint to resume from; the sibling *_training_state.pth "
                           "restores optimizer/scheduler/epoch automatically")
    ckpt.add_argument("--reset_optimizer", action="store_true",
                      help="Load only the weights from --resume (fine-tuning)")
    ckpt.add_argument("--best_metric", type=str, default="ms_ssim",
                      choices=["ms_ssim", "ssim", "psnr", "lpips", "val_loss"])
    ckpt.add_argument("--no_save_last", action="store_true",
                      help="Only keep the best checkpoint. Resuming then restarts from "
                           "the best epoch instead of the latest one")
    ckpt.add_argument("--early_stop_patience", type=int, default=0,
                      help="Stop after N validations without improvement; 0 disables it")

    log = parser.add_argument_group("logging")
    log.add_argument("--log_dir", type=str, default="./logs")
    log.add_argument("--log_interval", type=int, default=50,
                     help="Scalar logging interval in steps")
    log.add_argument("--image_log_interval", type=int, default=200,
                     help="Input/output/GT image logging interval in steps")
    log.add_argument("--num_log_images", type=int, default=4)
    log.add_argument("--val_interval", type=int, default=1, help="Validate every N epochs")
    log.add_argument("--no_val_lpips", action="store_true",
                     help="Skip LPIPS during validation")

    misc = parser.add_argument_group("misc")
    misc.add_argument("--seed", type=int, default=42)
    misc.add_argument("--deterministic", action="store_true",
                      help="Deterministic cudnn algorithms (slower)")
    misc.add_argument("--miopen_find_mode", type=str, default="FAST",
                      choices=["FAST", "NORMAL", "HYBRID", "DYNAMIC_HYBRID", "DEFAULT"],
                      help="ROCm only. MIOpen's convolution algorithm search mode. "
                           "NORMAL spends 5-8 minutes per new tensor shape on the "
                           "regular architecture; FAST costs seconds and measured as "
                           "fast or faster at steady state. DEFAULT leaves it unset")
    misc.add_argument("--cudnn_benchmark", type=str, default="auto",
                      choices=["auto", "on", "off"],
                      help="Autotune convolution algorithms. 'auto' enables it on CUDA "
                           "but not on ROCm, where it triggers an exhaustive MIOpen "
                           "search that can stall the first iterations for many minutes")
    misc.add_argument("--data_parallel", action="store_true",
                      help="Use nn.DataParallel in a single process instead of DDP. "
                           "Prefer torchrun + DDP; this is a convenience fallback")

    args = parser.parse_args()
    if args.val_batch_size is None:
        args.val_batch_size = args.batch_size
    height, width = args.resolution.lower().split("x")
    args.height, args.width = int(height), int(width)
    return args


# --------------------------------------------------------------------------- #
# Distributed helpers
# --------------------------------------------------------------------------- #
class DistributedEvalSampler(Sampler):
    """
    Strided, non-padding sampler for validation.

    ``DistributedSampler`` pads the last batch with duplicate samples, which
    would bias the validation metrics. Striding the indices keeps every sample
    exactly once at the cost of slightly uneven per-rank batch counts.
    """

    def __init__(self, dataset, rank, world_size):
        self.indices = list(range(rank, len(dataset), world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def setup_distributed(args):
    """Initialises the process group when launched through torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=60))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def all_reduce_sum(tensor):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def unwrap_model(model):
    """Peels off DDP/DataParallel and torch.compile wrappers."""
    while True:
        if hasattr(model, "module"):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
        else:
            return model


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(log_file, is_main):
    LOGGER.setLevel(logging.INFO if is_main else logging.WARNING)
    LOGGER.handlers.clear()
    LOGGER.propagate = False
    if not is_main:
        return
    formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    LOGGER.addHandler(stream)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def log_image_triplet(writer, tag, source, prediction, target, step, max_images):
    """Writes a source / prediction / ground-truth grid, one row per kind."""
    count = min(max_images, source.shape[0])
    rows = torch.cat(
        [
            source[:count].detach().float().cpu(),
            prediction[:count].detach().float().clamp(0, 1).cpu(),
            target[:count].detach().float().cpu(),
        ],
        dim=0,
    )
    grid = make_grid(rows, nrow=count, padding=2)
    writer.add_image(tag, grid, step)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_dataloaders(args, rank, world_size, distributed):
    common = dict(
        height=args.height,
        width=args.width,
        root=args.data_root,
        cache_source=not args.no_cache_source,
    )
    train_set = BlenderViewSynthesisDataset(
        args.train_file, hflip_prob=args.hflip_prob,
        max_samples=args.max_train_samples, **common,
    )
    val_set = BlenderViewSynthesisDataset(
        args.val_file, hflip_prob=0.0, max_samples=args.max_val_samples, **common,
    )

    if distributed:
        train_sampler = DistributedSampler(
            train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        val_sampler = DistributedEvalSampler(val_set, rank, world_size)
    else:
        train_sampler, val_sampler = None, None

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **loader_kwargs,
    )
    return train_set, val_set, train_loader, val_loader, train_sampler


def build_optimizer(args, model):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "soap":
        return SOAP(
            parameters,
            lr=args.lr,
            betas=tuple(args.betas),
            eps=args.eps,
            weight_decay=args.weight_decay,
            precondition_frequency=args.precondition_frequency,
            max_precond_dim=args.max_precond_dim,
            merge_dims=not args.no_merge_dims,
            precondition_1d=args.precondition_1d,
            update_clip=args.soap_update_clip,
            min_rank_ratio=args.soap_min_rank_ratio,
            data_format="channels_last" if args.channels_last else "channels_first",
        )
    return torch.optim.AdamW(
        parameters,
        lr=args.lr,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )


def build_val_metrics(args, device):
    """
    Returns (per-image metric functions, lpips module or None).

    The functional torchmetrics API is used on purpose: the stateful modules
    would accumulate one tensor per image for ``reduction='none'`` and would try
    to sync themselves across ranks, while here the reduction is done by hand.
    """
    from torchmetrics.functional.image import (
        multiscale_structural_similarity_index_measure,
        structural_similarity_index_measure,
    )

    metrics = {
        "ms_ssim": lambda pred, target: multiscale_structural_similarity_index_measure(
            pred, target, data_range=1.0, reduction="none"
        ),
        "ssim": lambda pred, target: structural_similarity_index_measure(
            pred, target, data_range=1.0, reduction="none"
        ),
    }
    lpips_fn = None
    if not args.no_val_lpips:
        try:
            import lpips as lpips_package

            lpips_fn = lpips_package.LPIPS(net="alex", verbose=False).to(device)
            lpips_fn.eval()
        except Exception as exc:
            LOGGER.warning("LPIPS unavailable (%s); validating without it.", exc)
    return metrics, lpips_fn


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def checkpoint_paths(args, prefix=""):
    """Mirrors the naming ``test.py`` parses: checkpoint_blender_pvsnet[_lite]_HxW.pth."""
    lite = "_lite" if args.architecture == "lite" else ""
    stem = f"{prefix}checkpoint_blender_pvsnet{lite}_{args.height}x{args.width}"
    directory = os.path.join(args.checkpoint_dir, args.run_name)
    return (
        os.path.join(directory, f"{stem}.pth"),
        os.path.join(directory, f"{stem}_training_state.pth"),
    )


def save_checkpoint(args, model, optimizer, scheduler, scaler, epoch, global_step,
                    best_value, metrics, prefix=""):
    weights_path, state_path = checkpoint_paths(args, prefix=prefix)
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    # Bare state_dict, so test.py / predict_view.py / the apps load it as-is.
    torch.save(unwrap_model(model).state_dict(), weights_path)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "best_value": best_value,
            "best_metric": args.best_metric,
            "metrics": metrics,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "args": vars(args),
        },
        state_path,
    )
    return weights_path, state_path


def load_resume(args, model, optimizer, scheduler, scaler, device):
    """Returns (start_epoch, global_step, best_value)."""
    weights = torch.load(args.resume, map_location="cpu", weights_only=False)
    if isinstance(weights, dict) and "state_dict" in weights:
        weights = weights["state_dict"]
    weights = {k.replace("module.", "", 1): v for k, v in weights.items()}
    missing, unexpected = unwrap_model(model).load_state_dict(weights, strict=False)
    if missing or unexpected:
        LOGGER.warning("Resume: missing=%s unexpected=%s", missing, unexpected)
    LOGGER.info("Loaded model weights from %s", args.resume)

    state_path = args.resume.replace(".pth", "_training_state.pth")
    if args.reset_optimizer:
        LOGGER.info("--reset_optimizer set: optimizer/scheduler start fresh")
        return 0, 0, None
    if not os.path.isfile(state_path):
        LOGGER.warning("No training state at %s; optimizer/scheduler start fresh",
                       state_path)
        return 0, 0, None

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    best_value = state.get("best_value")
    if state.get("best_metric") != args.best_metric:
        LOGGER.warning(
            "Checkpoint tracked '%s' but --best_metric is '%s'; resetting the best value",
            state.get("best_metric"), args.best_metric,
        )
        best_value = None
    start_epoch = int(state.get("epoch", -1)) + 1
    global_step = int(state.get("global_step", 0))
    LOGGER.info("Resuming at epoch %d (step %d), best %s = %s",
                start_epoch, global_step, args.best_metric, best_value)
    return start_epoch, global_step, best_value


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(args, model, val_loader, criterion, metrics, lpips_fn, device, writer,
             global_step, autocast_kwargs):
    model.eval()
    totals = {"val_loss": 0.0, "ms_ssim": 0.0, "ssim": 0.0, "psnr": 0.0, "lpips": 0.0}
    # Fixed order, so every rank reduces the same vector even when a rank ends
    # up with no samples at all.
    component_names = criterion.active_terms
    component_totals = {name: 0.0 for name in component_names}
    count = 0
    logged_images = False

    for batch in val_loader:
        source = batch["source"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        position = batch["position"].to(device, non_blocking=True)
        if args.channels_last:
            source = source.contiguous(memory_format=torch.channels_last)
            target = target.contiguous(memory_format=torch.channels_last)

        with torch.autocast(**autocast_kwargs):
            prediction = model(source, position)
        prediction = prediction.float().clamp(0.0, 1.0)
        target = target.float()

        loss, components = criterion(prediction, target)
        batch_size = source.shape[0]

        totals["val_loss"] += loss.item() * batch_size
        for name, value in components.items():
            component_totals[name] += value.item() * batch_size

        mse = (prediction - target).pow(2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
        totals["psnr"] += (10.0 * torch.log10(1.0 / mse)).sum().item()
        totals["ms_ssim"] += metrics["ms_ssim"](prediction, target).sum().item()
        totals["ssim"] += metrics["ssim"](prediction, target).sum().item()
        if lpips_fn is not None:
            totals["lpips"] += lpips_fn(
                prediction, target, normalize=True
            ).sum().item()
        count += batch_size

        if writer is not None and not logged_images:
            log_image_triplet(writer, "val/source_pred_gt", source, prediction, target,
                              global_step, args.num_log_images)
            logged_images = True

    stats = torch.tensor(
        [totals[k] for k in ("val_loss", "ms_ssim", "ssim", "psnr", "lpips")]
        + [component_totals[name] for name in component_names]
        + [float(count)],
        device=device,
        dtype=torch.float64,
    )
    all_reduce_sum(stats)
    total_count = max(1.0, stats[-1].item())

    results = {
        "val_loss": stats[0].item() / total_count,
        "ms_ssim": stats[1].item() / total_count,
        "ssim": stats[2].item() / total_count,
        "psnr": stats[3].item() / total_count,
    }
    if lpips_fn is not None:
        results["lpips"] = stats[4].item() / total_count
    for offset, name in enumerate(component_names):
        results[f"val_{name}"] = stats[5 + offset].item() / total_count

    model.train()
    return results


def is_improvement(metric_name, new_value, best_value):
    if best_value is None:
        return True
    if metric_name in HIGHER_IS_BETTER:
        return new_value > best_value
    return new_value < best_value


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    # Must be set before MIOpen sees its first convolution.
    if (
        torch.version.hip is not None
        and args.miopen_find_mode != "DEFAULT"
        and "MIOPEN_FIND_MODE" not in os.environ
    ):
        os.environ["MIOPEN_FIND_MODE"] = args.miopen_find_mode

    distributed, rank, world_size, local_rank = setup_distributed(args)
    is_main = rank == 0

    if args.run_name is None:
        # Resuming continues writing into the same run directory instead of
        # starting a new one, unless --run_name says otherwise.
        resumed_dir = (
            os.path.basename(os.path.dirname(os.path.abspath(args.resume)))
            if args.resume
            else None
        )
        if resumed_dir and resumed_dir != os.path.basename(
            os.path.abspath(args.checkpoint_dir)
        ):
            args.run_name = resumed_dir
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.run_name = (
                f"blender_{args.architecture}_{args.height}x{args.width}_{stamp}"
            )
    if distributed:
        # Every rank must agree on the run name, so broadcast rank 0's choice.
        name_holder = [args.run_name]
        dist.broadcast_object_list(name_holder, src=0)
        args.run_name = name_holder[0]

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    if args.cudnn_benchmark == "auto":
        use_benchmark = torch.version.hip is None
    else:
        use_benchmark = args.cudnn_benchmark == "on"
    torch.backends.cudnn.benchmark = use_benchmark and not args.deterministic
    torch.backends.cudnn.deterministic = args.deterministic

    run_log_dir = os.path.join(args.log_dir, args.run_name)
    if is_main:
        os.makedirs(run_log_dir, exist_ok=True)
        os.makedirs(os.path.join(args.checkpoint_dir, args.run_name), exist_ok=True)
    if distributed:
        dist.barrier()

    setup_logging(os.path.join(run_log_dir, "train_log.txt"), is_main)
    writer = SummaryWriter(os.path.join(run_log_dir, "tensorboard")) if is_main else None

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if is_main:
        LOGGER.info("Run: %s", args.run_name)
        LOGGER.info(
            "Device: %s (%s) | torch %s",
            device,
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            torch.__version__,
        )
        LOGGER.info("Arguments: %s", json.dumps(vars(args), indent=2, default=str))
        with open(os.path.join(run_log_dir, "args.json"), "w") as handle:
            json.dump(vars(args), handle, indent=2, default=str)

    # --- data ---
    train_set, val_set, train_loader, val_loader, train_sampler = build_dataloaders(
        args, rank, world_size, distributed
    )
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum_steps)
    global_batch = args.batch_size * world_size * args.grad_accum_steps
    LOGGER.info(
        "Train samples: %d | Val samples: %d | GPUs: %d | global batch: %d | steps/epoch: %d",
        len(train_set), len(val_set), world_size, global_batch, steps_per_epoch,
    )

    # --- model ---
    model = PVSNet(
        total_image_input=params.params_number_input,
        pose_dims=args.pose_dims,
        height=args.height,
        width=args.width,
        is_lite=args.architecture == "lite",
    ).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    LOGGER.info(
        "Model parameters: %.2fM",
        sum(p.numel() for p in model.parameters()) / 1e6,
    )

    # --- loss ---
    # The VGG/DINO/LPIPS backbones are fetched into a shared torch.hub cache, so
    # let rank 0 populate it alone before the other ranks read from it.
    if distributed and rank != 0:
        dist.barrier()
    criterion = CombinedLoss(
        w_msssim_l1=args.w_msssim_l1,
        w_l2=args.w_l2,
        w_ffl=args.w_ffl,
        w_vgg=args.w_vgg,
        w_dino=args.w_dino,
        msssim_alpha=args.msssim_alpha,
        ffl_alpha=args.ffl_alpha,
        ffl_patch_factor=args.ffl_patch_factor,
        vgg_resize=args.vgg_resize,
        dino_model=args.dino_model,
        dino_input_size=args.dino_input_size,
    ).to(device)
    metrics, lpips_fn = build_val_metrics(args, device)
    if distributed and rank == 0:
        dist.barrier()
    LOGGER.info("Active loss terms: %s", ", ".join(criterion.active_terms))

    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(
        optimizer,
        kind=args.scheduler,
        steps_per_epoch=steps_per_epoch,
        total_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        step_size_epochs=args.step_size_epochs,
        gamma=args.gamma,
        min_lr_ratio=args.min_lr_ratio,
    )

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    if amp_dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        LOGGER.warning("bf16 unsupported on this GPU; falling back to fp16")
        amp_dtype = torch.float16
        args.amp = "fp16"
    amp_enabled = amp_dtype is not None and device.type == "cuda"
    autocast_kwargs = dict(
        device_type=device.type,
        dtype=amp_dtype if amp_enabled else torch.bfloat16,
        enabled=amp_enabled,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp == "fp16")

    start_epoch, global_step, best_value = 0, 0, None
    if args.resume:
        start_epoch, global_step, best_value = load_resume(
            args, model, optimizer, scheduler, scaler, device
        )

    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None
        )
    elif args.data_parallel and torch.cuda.device_count() > 1:
        LOGGER.info("Using nn.DataParallel over %d GPUs", torch.cuda.device_count())
        model = nn.DataParallel(model)
    if args.compile:
        model = torch.compile(model)

    model.train()
    epochs_without_improvement = 0
    skipped_steps = 0
    LOGGER.info("Starting training at epoch %d for %d epochs", start_epoch, args.epochs)

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_batches = 0
        window_start = time.time()
        window_images = 0
        last_grad_norm = float("nan")
        optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(train_loader):
            source = batch["source"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            position = batch["position"].to(device, non_blocking=True)
            if args.channels_last:
                source = source.contiguous(memory_format=torch.channels_last)
                target = target.contiguous(memory_format=torch.channels_last)

            is_accum_boundary = (batch_index + 1) % args.grad_accum_steps == 0
            sync_context = (
                model.no_sync()
                if distributed and not is_accum_boundary
                else contextlib.nullcontext()
            )

            with sync_context:
                with torch.autocast(**autocast_kwargs):
                    prediction = model(source, position)
                # Losses run in fp32: the FFT and the frozen backbones are
                # sensitive to reduced precision.
                prediction = prediction.float()
                loss, components = criterion(prediction, target.float())
                loss_is_finite = torch.isfinite(loss).item()
                if loss_is_finite:
                    scaler.scale(loss / args.grad_accum_steps).backward()

            if not loss_is_finite:
                # Backward is skipped so .grad never sees the bad batch.
                optimizer.zero_grad(set_to_none=True)
                skipped_steps += 1
                LOGGER.warning("Non-finite loss at step %d; skipping the update",
                               global_step + 1)
            elif is_accum_boundary:
                parameters = unwrap_model(model).parameters()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
                else:
                    grad_norm = torch.nn.utils.get_total_norm(
                        [p.grad for p in parameters if p.grad is not None]
                    )
                # A single inf/nan gradient would otherwise be spread across every
                # parameter by the clipping rescale, which makes the weights NaN
                # for good. Under bf16 there is no GradScaler to catch that.
                if torch.isfinite(grad_norm):
                    scaler.step(optimizer)
                    scaler.update()
                    skipped_steps = 0
                else:
                    skipped_steps += 1
                    LOGGER.warning(
                        "Non-finite gradient norm at step %d; skipping the update",
                        global_step + 1,
                    )
                last_grad_norm = grad_norm.item()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            if args.skip_limit > 0 and skipped_steps >= args.skip_limit:
                raise RuntimeError(
                    f"{skipped_steps} consecutive non-finite updates at step "
                    f"{global_step}. The run has diverged; lower --lr (or raise "
                    "--warmup_steps) and restart from the last good checkpoint."
                )

            if loss_is_finite:
                epoch_loss += loss.item()
                epoch_batches += 1
            window_images += source.shape[0] * world_size

            if is_main and is_accum_boundary and global_step % args.log_interval == 0:
                elapsed = max(1e-6, time.time() - window_start)
                throughput = window_images / elapsed
                writer.add_scalar("train/loss", loss.item(), global_step)
                for name, value in components.items():
                    writer.add_scalar(f"train/{name}", value.item(), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("train/grad_norm", last_grad_norm, global_step)
                writer.add_scalar("train/images_per_sec", throughput, global_step)
                peak_memory = (
                    torch.cuda.max_memory_allocated(device) / 1e9
                    if device.type == "cuda"
                    else 0.0
                )
                writer.add_scalar("train/peak_mem_gb", peak_memory, global_step)
                component_text = " ".join(
                    f"{name}={value.item():.4f}" for name, value in components.items()
                )
                LOGGER.info(
                    "epoch %d/%d step %d (%d/%d) | loss %.4f | %s | lr %.2e"
                    " | grad %.3f | %.1f img/s | peak %.1fGB",
                    epoch + 1, args.epochs, global_step, batch_index + 1,
                    len(train_loader), loss.item(), component_text,
                    scheduler.get_last_lr()[0], last_grad_norm, throughput, peak_memory,
                )
                window_start = time.time()
                window_images = 0

            if (
                is_main
                and is_accum_boundary
                and args.image_log_interval > 0
                and global_step % args.image_log_interval == 0
            ):
                log_image_triplet(writer, "train/source_pred_gt", source, prediction,
                                  target, global_step, args.num_log_images)

        mean_epoch_loss = epoch_loss / max(1, epoch_batches)
        if is_main:
            writer.add_scalar("train/epoch_loss", mean_epoch_loss, epoch + 1)
        LOGGER.info("epoch %d finished | mean train loss %.4f", epoch + 1,
                    mean_epoch_loss)

        if (epoch + 1) % args.val_interval != 0 and (epoch + 1) != args.epochs:
            continue

        results = validate(args, model, val_loader, criterion, metrics, lpips_fn,
                           device, writer, global_step, autocast_kwargs)
        if is_main:
            for name, value in results.items():
                writer.add_scalar(f"val/{name}", value, global_step)
            LOGGER.info(
                "validation epoch %d | %s",
                epoch + 1,
                " | ".join(f"{name} {value:.4f}" for name, value in results.items()),
            )

        tracked = results.get(args.best_metric)
        if tracked is None:
            LOGGER.warning("Metric '%s' is unavailable; tracking val_loss instead",
                           args.best_metric)
            args.best_metric = "val_loss"
            tracked = results["val_loss"]

        improved = is_improvement(args.best_metric, tracked, best_value)
        if improved:
            previous, best_value = best_value, tracked
            epochs_without_improvement = 0
            if is_main:
                weights_path, _ = save_checkpoint(
                    args, model, optimizer, scheduler, scaler, epoch, global_step,
                    best_value, results,
                )
                LOGGER.info(
                    "%s improved %s -> %.4f; saved %s",
                    args.best_metric,
                    "n/a" if previous is None else f"{previous:.4f}",
                    best_value, weights_path,
                )
        else:
            epochs_without_improvement += 1
            LOGGER.info("%s did not improve (best %.4f, now %.4f) | %d validation(s) without improvement",
                        args.best_metric, best_value, tracked,
                        epochs_without_improvement)

        if is_main and not args.no_save_last:
            # Written every validation so an interrupted run resumes from the
            # latest epoch rather than the last improvement.
            save_checkpoint(args, model, optimizer, scheduler, scaler, epoch,
                            global_step, best_value, results, prefix="last_")

        if distributed:
            dist.barrier()

        if (
            args.early_stop_patience > 0
            and epochs_without_improvement >= args.early_stop_patience
        ):
            LOGGER.info("Early stopping after %d validations without improvement",
                        epochs_without_improvement)
            break

    if is_main:
        LOGGER.info("Training finished. Best %s = %s", args.best_metric, best_value)
        writer.close()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

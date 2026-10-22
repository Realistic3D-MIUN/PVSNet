"""
Learning-rate schedules for ``train.py``.

Both schedules are expressed as per-optimizer-step lambdas (rather than
per-epoch) so that warmup is smooth and so resuming mid-run restores the exact
same learning rate. ``step_size_epochs`` is still specified in epochs and
converted internally.
"""

import math

from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(optimizer, kind="step", steps_per_epoch=1, total_epochs=100,
                    warmup_steps=0, step_size_epochs=30, gamma=0.5,
                    min_lr_ratio=0.01):
    """
    Args:
        kind: ``step`` for StepLR-style decay, ``cosine`` for cosine annealing.
        steps_per_epoch: Optimizer steps per epoch (after gradient accumulation).
        total_epochs: Total epochs, used to size the cosine schedule.
        warmup_steps: Linear warmup length in optimizer steps.
        step_size_epochs: Decay period in epochs (``step`` only).
        gamma: Multiplicative decay factor (``step`` only).
        min_lr_ratio: Floor as a fraction of the base LR (``cosine`` only).

    Returns:
        A ``LambdaLR`` that must be stepped once per optimizer step.
    """
    steps_per_epoch = max(1, int(steps_per_epoch))
    total_steps = max(1, steps_per_epoch * int(total_epochs))
    decay_steps = max(1, steps_per_epoch * int(step_size_epochs))

    if kind == "step":
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            return gamma ** ((step - warmup_steps) // decay_steps)
    elif kind == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, progress)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    else:
        raise ValueError(f"Unknown scheduler '{kind}', expected 'step' or 'cosine'")

    return LambdaLR(optimizer, lr_lambda)

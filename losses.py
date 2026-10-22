"""
Loss functions used by ``train.py``.

Everything operates on RGB tensors of shape (N, 3, H, W) in the [0, 1] range,
which is what PVSNet outputs (its last layer is a sigmoid) and what the data
loader produces for the ground truth.

MS-SSIM and the focal frequency loss are implemented here directly so that
training only needs torch/torchvision. The VGG and DINO terms lazily download
their backbones the first time they are used, and are skipped entirely when
their weight is zero.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Scale weights from Wang et al., "Multiscale structural similarity for image
# quality assessment" (2003).
MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _gaussian_kernel1d(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype)
    coords -= (window_size - 1) / 2.0
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _separable_filter(x: torch.Tensor, kernel1d: torch.Tensor) -> torch.Tensor:
    """Applies a 1D kernel along H and then W, without padding (valid conv)."""
    channels = x.shape[1]
    k = kernel1d.to(device=x.device, dtype=x.dtype)
    row = k.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    col = k.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    x = F.conv2d(x, row, groups=channels)
    x = F.conv2d(x, col, groups=channels)
    return x


def _ssim_maps(pred, target, kernel1d, data_range=1.0, k1=0.01, k2=0.03):
    """Returns the per-channel mean SSIM and contrast-sensitivity terms."""
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mu_pred = _separable_filter(pred, kernel1d)
    mu_target = _separable_filter(target, kernel1d)

    mu_pred_sq = mu_pred * mu_pred
    mu_target_sq = mu_target * mu_target
    mu_pred_target = mu_pred * mu_target

    sigma_pred = _separable_filter(pred * pred, kernel1d) - mu_pred_sq
    sigma_target = _separable_filter(target * target, kernel1d) - mu_target_sq
    sigma_cross = _separable_filter(pred * target, kernel1d) - mu_pred_target

    cs = (2 * sigma_cross + c2) / (sigma_pred + sigma_target + c2)
    ssim = ((2 * mu_pred_target + c1) / (mu_pred_sq + mu_target_sq + c1)) * cs
    return ssim.mean(dim=(2, 3)), cs.mean(dim=(2, 3))


def ms_ssim(pred, target, window_size=11, sigma=1.5, data_range=1.0, weights=None):
    """
    Differentiable MS-SSIM averaged over the batch.

    The number of scales is reduced automatically for small inputs so that the
    coarsest scale still fits the Gaussian window.
    """
    weights = list(MS_SSIM_WEIGHTS if weights is None else weights)
    smallest_side = min(pred.shape[-2:])
    max_levels = int(math.floor(math.log2(smallest_side / (window_size - 1)))) + 1
    levels = max(1, min(len(weights), max_levels))
    if levels < len(weights):
        weights = weights[:levels]
        total = sum(weights)
        weights = [w / total for w in weights]

    kernel1d = _gaussian_kernel1d(window_size, sigma, pred.device, pred.dtype)

    cs_values = []
    ssim_value = None
    for level in range(levels):
        ssim_value, cs = _ssim_maps(pred, target, kernel1d, data_range=data_range)
        if level < levels - 1:
            cs_values.append(F.relu(cs))
            pad_h = pred.shape[-2] % 2
            pad_w = pred.shape[-1] % 2
            if pad_h or pad_w:
                pred = F.pad(pred, (0, pad_w, 0, pad_h), mode="replicate")
                target = F.pad(target, (0, pad_w, 0, pad_h), mode="replicate")
            pred = F.avg_pool2d(pred, kernel_size=2)
            target = F.avg_pool2d(target, kernel_size=2)

    cs_values.append(F.relu(ssim_value))
    stacked = torch.stack(cs_values, dim=0)  # (levels, N, C)
    weight_tensor = torch.tensor(weights, device=pred.device, dtype=pred.dtype)
    ms = torch.prod(stacked ** weight_tensor.view(-1, 1, 1), dim=0)
    return ms.mean()


class MSSSIML1Loss(nn.Module):
    """
    The MS-SSIM + L1 mix from Zhao et al., "Loss Functions for Image Restoration
    with Neural Networks" (2017):  alpha * (1 - MS-SSIM) + (1 - alpha) * G_sigma * L1.

    The L1 term is weighted by a Gaussian of the coarsest MS-SSIM scale, which is
    what makes the two terms agree about which pixels matter.
    """

    def __init__(self, alpha=0.84, window_size=11, sigma=1.5, gaussian_l1=True,
                 data_range=1.0):
        super().__init__()
        self.alpha = alpha
        self.window_size = window_size
        self.sigma = sigma
        self.gaussian_l1 = gaussian_l1
        self.data_range = data_range
        # sigma of the coarsest scale, as in the paper.
        self.l1_sigma = sigma * (2 ** (len(MS_SSIM_WEIGHTS) - 1))
        self.l1_window = int(2 * math.ceil(3 * self.l1_sigma) + 1)

    def forward(self, pred, target):
        ms = ms_ssim(
            pred,
            target,
            window_size=self.window_size,
            sigma=self.sigma,
            data_range=self.data_range,
        )
        l1_map = torch.abs(pred - target)
        if self.gaussian_l1 and min(pred.shape[-2:]) > self.l1_window:
            kernel = _gaussian_kernel1d(
                self.l1_window, self.l1_sigma, pred.device, pred.dtype
            )
            l1_term = _separable_filter(l1_map, kernel).mean()
        else:
            l1_term = l1_map.mean()
        return self.alpha * (1.0 - ms) + (1.0 - self.alpha) * l1_term


class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss (Jiang et al., ICCV 2021).

    Penalises the squared distance between the 2D DFTs of prediction and target,
    re-weighted by a detached, normalised version of that same distance so the
    hard frequencies dominate. This is what recovers the high-frequency detail
    that pixel losses blur away.
    """

    def __init__(self, alpha=1.0, patch_factor=1, ave_spectrum=False,
                 log_matrix=False, batch_matrix=False):
        super().__init__()
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def _to_patch_spectrum(self, x):
        pf = self.patch_factor
        h, w = x.shape[-2:]
        if pf == 1:
            patches = x.unsqueeze(1)
        else:
            assert h % pf == 0 and w % pf == 0, (
                f"patch_factor {pf} does not divide image size {h}x{w}"
            )
            ph, pw = h // pf, w // pf
            patches = (
                x.unfold(2, ph, ph)
                .unfold(3, pw, pw)
                .permute(0, 2, 3, 1, 4, 5)
                .reshape(x.shape[0], pf * pf, x.shape[1], ph, pw)
            )
        freq = torch.fft.fft2(patches.float(), norm="ortho")
        return torch.stack([freq.real, freq.imag], dim=-1)

    def forward(self, pred, target):
        freq_pred = self._to_patch_spectrum(pred)
        freq_target = self._to_patch_spectrum(target)
        if self.ave_spectrum:
            freq_pred = freq_pred.mean(dim=0, keepdim=True)
            freq_target = freq_target.mean(dim=0, keepdim=True)

        distance = (freq_pred - freq_target).pow(2).sum(dim=-1)

        with torch.no_grad():
            weight = distance.detach() ** self.alpha
            if self.log_matrix:
                weight = torch.log(weight + 1.0)
            if self.batch_matrix:
                weight = weight / weight.max().clamp_min(1e-12)
            else:
                flat_max = weight.flatten(1).max(dim=1).values
                weight = weight / flat_max.clamp_min(1e-12).view(-1, 1, 1, 1, 1)
            weight = torch.nan_to_num(weight, nan=0.0).clamp(0.0, 1.0)

        return (weight * distance).mean()


class VGGPerceptualLoss(nn.Module):
    """L1 distance between frozen ImageNet VGG19 feature maps."""

    LAYER_IDS = {"relu1_2": 3, "relu2_2": 8, "relu3_4": 17, "relu4_4": 26}

    def __init__(self, layers=("relu1_2", "relu2_2", "relu3_4", "relu4_4"),
                 layer_weights=None, resize_to=None):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19

        unknown = [layer for layer in layers if layer not in self.LAYER_IDS]
        if unknown:
            raise ValueError(f"Unknown VGG layers: {unknown}")

        self.layer_ids = [self.LAYER_IDS[layer] for layer in layers]
        self.layer_weights = (
            [1.0] * len(layers) if layer_weights is None else list(layer_weights)
        )
        self.resize_to = resize_to

        features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.features = features[: max(self.layer_ids) + 1].eval()
        for param in self.features.parameters():
            param.requires_grad_(False)

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode=True):
        # The backbone stays frozen in eval mode regardless of the parent module.
        super().train(mode)
        self.features.eval()
        return self

    def _prepare(self, x):
        if self.resize_to is not None and x.shape[-1] != self.resize_to:
            x = F.interpolate(
                x, size=(self.resize_to, self.resize_to), mode="bilinear",
                align_corners=False, antialias=True,
            )
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        pred = self._prepare(pred)
        target = self._prepare(target)
        loss = pred.new_zeros(())
        idx = 0
        for layer_id, layer in enumerate(self.features):
            pred = layer(pred)
            target = layer(target)
            if layer_id in self.layer_ids:
                loss = loss + self.layer_weights[idx] * F.l1_loss(pred, target)
                idx += 1
                if idx == len(self.layer_ids):
                    break
        return loss / sum(self.layer_weights)


class DINOLoss(nn.Module):
    """
    Self-supervised feature loss on a frozen DINOv2 ViT.

    Combines a cosine distance on the CLS token (global/semantic agreement) with
    an L1 distance on the patch tokens (local layout agreement). DINO features
    are less texture-biased than VGG, which helps view synthesis keep structures
    in the right place.
    """

    def __init__(self, model_name="dinov2_vits14", input_size=224,
                 patch_weight=1.0, cls_weight=1.0):
        super().__init__()
        try:
            # skip_validation avoids a GitHub API call that is easy to rate-limit
            # on shared machines; the weights still come from the official repo.
            backbone = torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
                verbose=False,
                trust_repo=True,
                skip_validation=True,
            )
        except Exception as exc:  # pragma: no cover - depends on network access
            raise RuntimeError(
                f"Could not load DINO backbone '{model_name}'. Pre-download it with "
                "torch.hub, or disable this term with --w_dino 0."
            ) from exc

        patch_size = getattr(backbone, "patch_size", 14)
        if input_size % patch_size != 0:
            input_size = max(patch_size, round(input_size / patch_size) * patch_size)
        self.input_size = input_size
        self.patch_weight = patch_weight
        self.cls_weight = cls_weight

        self.backbone = backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _prepare(self, x):
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(
                x, size=(self.input_size, self.input_size), mode="bilinear",
                align_corners=False, antialias=True,
            )
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        features_pred = self.backbone.forward_features(self._prepare(pred))
        features_target = self.backbone.forward_features(self._prepare(target))

        cls_pred = features_pred["x_norm_clstoken"]
        cls_target = features_target["x_norm_clstoken"]
        cls_loss = (1.0 - F.cosine_similarity(cls_pred, cls_target, dim=-1)).mean()

        patch_loss = F.l1_loss(
            features_pred["x_norm_patchtokens"], features_target["x_norm_patchtokens"]
        )
        return self.cls_weight * cls_loss + self.patch_weight * patch_loss


class CombinedLoss(nn.Module):
    """
    Weighted sum of every enabled term.

    Terms with a weight of zero are never constructed, so disabling the VGG or
    DINO term also avoids downloading and holding its backbone.
    """

    def __init__(self, w_msssim_l1=1.0, w_l2=0.1, w_ffl=1.0, w_vgg=0.1, w_dino=0.05,
                 msssim_alpha=0.84, ffl_alpha=1.0, ffl_patch_factor=1,
                 vgg_resize=None, dino_model="dinov2_vits14", dino_input_size=224):
        super().__init__()
        self.weights = {
            "msssim_l1": w_msssim_l1,
            "l2": w_l2,
            "ffl": w_ffl,
            "vgg": w_vgg,
            "dino": w_dino,
        }

        self.msssim_l1 = (
            MSSSIML1Loss(alpha=msssim_alpha) if w_msssim_l1 > 0 else None
        )
        self.ffl = (
            FocalFrequencyLoss(alpha=ffl_alpha, patch_factor=ffl_patch_factor)
            if w_ffl > 0
            else None
        )
        self.vgg = VGGPerceptualLoss(resize_to=vgg_resize) if w_vgg > 0 else None
        self.dino = (
            DINOLoss(model_name=dino_model, input_size=dino_input_size)
            if w_dino > 0
            else None
        )

    @property
    def active_terms(self):
        return [name for name, weight in self.weights.items() if weight > 0]

    def forward(self, pred, target):
        """Returns (total_loss, {term_name: unweighted_value})."""
        components = {}

        if self.msssim_l1 is not None:
            components["msssim_l1"] = self.msssim_l1(pred, target)
        if self.weights["l2"] > 0:
            components["l2"] = F.mse_loss(pred, target)
        if self.ffl is not None:
            components["ffl"] = self.ffl(pred, target)
        if self.vgg is not None:
            components["vgg"] = self.vgg(pred, target)
        if self.dino is not None:
            components["dino"] = self.dino(pred, target)

        total = pred.new_zeros(())
        for name, value in components.items():
            total = total + self.weights[name] * value
        return total, components

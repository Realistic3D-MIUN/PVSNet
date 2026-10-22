"""
Dataset for the Blender view-synthesis split files (``lf_train/validate/test.txt``).

Each line is::

    source_image 0 0 0 target_image target_x target_y target_z

The source is always the scene's centre view and the target is the view rendered
after translating the camera by (x, y, z), so the model learns to map
(centre view, translation) -> translated view.

Only the paths are held in memory; images are decoded in ``__getitem__`` by the
worker processes. Because a whole scene shares one centre view (58 distinct
source images for ~8.9k training pairs), decoded sources are cached per worker
as uint8 tensors, which removes most of the decode work.
"""

import os
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def normalize_position(x, y, z, pos_min=-0.1, pos_max=0.1):
    """
    Maps a translation in metres to the [0, 1] range the model was trained with.

    Mirrors ``getPositionVector`` in ``test.py`` (including its 7-decimal
    rounding) so training and evaluation agree exactly.
    """
    span = pos_max - pos_min
    vector = torch.zeros((1, 3), dtype=torch.float)
    vector[0][0] = (float(format(x, ".7f")) - pos_min) / span
    vector[0][1] = (float(format(y, ".7f")) - pos_min) / span
    vector[0][2] = (float(format(z, ".7f")) - pos_min) / span
    return vector


class BlenderViewSynthesisDataset(Dataset):
    """
    Args:
        list_file: Path to an ``lf_*.txt`` split file.
        height, width: Output resolution; images are resized on load.
        root: Directory the (relative) paths inside the split file are resolved
            against. Defaults to the directory containing the split file.
        pos_min, pos_max: Translation range used for position normalisation.
        cache_source: Keep decoded, resized source images in RAM (per worker).
        max_source_cache: Upper bound on cached source images.
        hflip_prob: Probability of flipping source and target horizontally while
            negating the x translation. Off by default because it assumes the
            x axis points along the image width.
        max_samples: Truncate the split, useful for smoke tests.
    """

    def __init__(self, list_file, height=256, width=256, root=None, pos_min=-0.1,
                 pos_max=0.1, cache_source=True, max_source_cache=128, hflip_prob=0.0,
                 max_samples=None):
        super().__init__()
        self.list_file = list_file
        self.height = height
        self.width = width
        self.root = root if root is not None else os.path.dirname(
            os.path.abspath(list_file)
        )
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.cache_source = cache_source
        self.max_source_cache = max_source_cache
        self.hflip_prob = hflip_prob

        rows = np.loadtxt(list_file, dtype=str)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if max_samples is not None:
            rows = rows[:max_samples]

        self.source_paths = [self._resolve(p) for p in rows[:, 0]]
        self.target_paths = [self._resolve(p) for p in rows[:, 4]]
        self.positions = rows[:, 5:8].astype(np.float64)

        self.resize = transforms.Resize((height, width))
        self._source_cache = OrderedDict()

    def _resolve(self, path):
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.root, path))

    def __len__(self):
        return len(self.source_paths)

    def _load_uint8(self, path):
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != (self.width, self.height):
                image = self.resize(image)
            return torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1)

    def _load_source(self, path):
        if not self.cache_source:
            return self._load_uint8(path)
        cached = self._source_cache.get(path)
        if cached is None:
            cached = self._load_uint8(path)
            self._source_cache[path] = cached
            if len(self._source_cache) > self.max_source_cache:
                self._source_cache.popitem(last=False)
        else:
            self._source_cache.move_to_end(path)
        return cached

    def __getitem__(self, index):
        source = self._load_source(self.source_paths[index]).float().div_(255.0)
        target = self._load_uint8(self.target_paths[index]).float().div_(255.0)
        x, y, z = self.positions[index]

        if self.hflip_prob > 0 and torch.rand(()) < self.hflip_prob:
            source = torch.flip(source, dims=[-1])
            target = torch.flip(target, dims=[-1])
            x = -x

        return {
            "source": source,
            "target": target,
            "position": normalize_position(x, y, z, self.pos_min, self.pos_max),
            "target_path": self.target_paths[index],
        }

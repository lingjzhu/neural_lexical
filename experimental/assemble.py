import re

out_file = "cached_contrastive.py"

with open("../pylate/pylate/utils/distributed.py", "r") as f:
    dist_code = f.read().split("import torch.distributed as dist")[1]

with open("../pylate/pylate/utils/tensor.py", "r") as f:
    tensor_code = f.read().split("import torch")[1]

with open("../pylate/pylate/scores/scores.py", "r") as f:
    scores_code = f.read()
    # Extract only colbert_scores
    match = re.search(r"def colbert_scores\(.*?return scores", scores_code, re.DOTALL)
    if match:
        scores_code = match.group(0)

with open("../pylate/pylate/losses/cached_contrastive.py", "r") as f:
    cached_code = f.read()
    # Remove relative imports
    cached_code = re.sub(r"from \.\..*?\n", "", cached_code)
    cached_code = re.sub(r"from \.contrastive.*?\n", "", cached_code)
    # Remove ColBERT type hints
    cached_code = cached_code.replace("model: ColBERT,", "model: nn.Module,")
    # Take from class RandContext onwards
    cached_code = "class RandContext:" + cached_code.split("class RandContext:")[1]

assembled = f"""from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import nullcontext
from functools import partial
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import tqdm
from torch import Tensor
from torch.utils.checkpoint import get_device_states, set_device_states

# --- From pylate.utils.distributed ---
{dist_code}

# --- From pylate.utils.tensor ---
{tensor_code}

# --- From pylate.scores.scores ---
{scores_code}

# --- From pylate.losses.cached_contrastive ---
{cached_code}
"""

with open(out_file, "w") as f:
    f.write(assembled)

# Dataset of sin function

from abc import ABC
from typing import Any, Dict, List

import numpy as np
from torch.utils.data import Dataset


class EqxDataset(Dataset, ABC):
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(batch[0]["input"], np.ndarray):
            inputs = np.stack([sample["input"] for sample in batch])
            targets = np.array([sample["target"] for sample in batch])
        else:
            inputs = np.stack([sample["input"].numpy() for sample in batch])
            targets = np.array([sample["target"] for sample in batch])
        return {"input": inputs, "target": targets}

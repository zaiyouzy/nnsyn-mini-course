"""The trainer class used for the reported 300-epoch baseline."""
from __future__ import annotations

import torch

from nnunetv2.training.nnUNetTrainer.variants.nnsyn.nnUNetTrainer_nnsyn_loss_masked import (
    nnUNetTrainer_nnsyn_loss_masked,
)


class nnUNetTrainer_nnsyn_loss_masked_300epochs(nnUNetTrainer_nnsyn_loss_masked):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans,
            configuration,
            fold,
            dataset_json,
            unpack_dataset,
            device,
        )
        self.num_epochs = 300
        self.save_every = 50

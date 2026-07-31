import torch

from nnunetv2.training.nnUNetTrainer.variants.nnsyn.nnUNetTrainer_nnsyn import (
    nnUNetTrainer_nnsyn,
)


class nnUNetTrainer_nnsyn_smoke(nnUNetTrainer_nnsyn):
    """Very short course-only run that validates the complete training path."""

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
        self.num_epochs = 1
        self.num_iterations_per_epoch = 2
        self.num_val_iterations_per_epoch = 1

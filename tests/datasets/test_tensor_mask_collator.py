import torch
from torch.utils.data import DataLoader

from src.masks.multiseq_multiblock3d import TensorMaskCollator

class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        clip = torch.zeros(3, 4, 32, 32)
        clip[:, :2, :16, :16] = 1.0
        valid = torch.tensor([0], dtype=torch.int64)
        return [clip], 0, [torch.arange(4)], [valid]

def test_masks_respect_valid_indices():
    cfg = [{
        "spatial_scale": (0.5, 0.5),
        "temporal_scale": (1.0, 1.0),
        "aspect_ratio": (1.0, 1.0),
        "num_blocks": 1,
    }]
    collator = TensorMaskCollator(
        cfgs_mask=cfg,
        dataset_fpcs=[4],
        crop_size=(32, 32),
        patch_size=(16, 16),
        tubelet_size=2,
    )
    loader = DataLoader(DummyDataset(), batch_size=1, collate_fn=collator)
    batch = next(iter(loader))
    collated_batch, masks_encs, masks_preds = batch[0]
    valid = collated_batch[3][0]
    for tensor in masks_encs + masks_preds:
        assert set(tensor[0].tolist()).issubset(set(valid.tolist()))

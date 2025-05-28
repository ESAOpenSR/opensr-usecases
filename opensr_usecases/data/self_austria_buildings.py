import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path


class AustriaDataset(Dataset):
    # This dataset is built for loading pre-inferred masks and gt data
    def __init__(self, data_path, inferred_data_path, pred_folder, run, gt_folder='gt', num_images=None, phase="test", image_type="lr"):
        self.height = 512
        self.width = 512
        self.num_images = num_images
        self.data = pd.read_csv(data_path)
        self.inferred_data_path = inferred_data_path
        self.pred_folder = pred_folder
        self.gt_folder = gt_folder
        self.run = run

        if self.pred_folder == 'orthofoto':
            # assume hr is orthophoto
            self.id_prefix = 'HR_ortho'
        else:
            self.id_prefix = 'S2'
    def __len__(self):
        return self.num_images if self.num_images is not None else len(self.data)

    def __getitem__(self, idx):
        pred_id = f"{self.id_prefix}_{self.data.loc[idx, 'id']:05d}.tif"
        mask_id = f"HR_mask_{self.data.loc[idx, 'id']:05d}.tif"

        pred_path = str(Path(self.inferred_data_path) / self.pred_folder / self.run / 'imgs' / pred_id)
        gt_path = str(Path(self.inferred_data_path) / self.gt_folder / mask_id)

        # Load predicted
        with rasterio.open(pred_path) as src:
            pred = src.read(1).astype(np.uint8)

        # Load gt mask
        with rasterio.open(gt_path) as src:
            mask = src.read(1).astype(np.uint8)

        return pred, mask


if __name__ == "__main__":
    # Example usage:
    dataset = AustriaDataset(data_path=Path(r'C:\Users\PC\Desktop\TU\Master\MasterThesis\data\stratification_tables\test.csv'), num_images=100)

    # To retrieve an image and mask
    image, mask = dataset[0]

    # Visualize the image
    viz = False
    if viz:
        image_np = image.permute(1, 2, 0).numpy()  # Convert back to HWC format for visualization
        plt.imshow(image_np)
        plt.title('Random 4-Band Image with Random Squares')
        plt.savefig("a.png")

        # Visualize the binary mask
        plt.imshow(mask.numpy(), cmap='gray')
        plt.title('Binary Mask for the Squares')
        plt.savefig("b.png")

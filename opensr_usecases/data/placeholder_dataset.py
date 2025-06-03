import torch
from torch.utils.data import Dataset
import numpy as np
import random
import matplotlib.pyplot as plt

class PlaceholderDataset(Dataset):
    def __init__(self, num_images=250,phase="test", image_type="lr"):
        self.height = 512
        self.width = 512
        self.num_images = num_images

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # Create a random 4-band image
        image = np.random.randint(0, 256, (self.height, self.width, 4), dtype=np.uint8)

        # Create a binary mask initialized to zeros
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        for _ in range(25):
            target_area = random.randint(1, 1225)  # pixel area between 1 and 35x35
            side = max(1, int(np.sqrt(target_area)))  # ensure side length >= 1

            # Constrain to image boundaries
            if side >= self.width or side >= self.height:
                continue

            x = random.randint(0, self.width - side)
            y = random.randint(0, self.height - side)

            # Assign random color for the square (one for each band)
            color = np.random.randint(0, 256, (4,), dtype=np.uint8)

            # Draw the square on image and mask
            image[y:y+side, x:x+side] = color
            mask[y:y+side, x:x+side] = 1

        # Convert the image and mask to tensors
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image_tensor, mask_tensor


if __name__ == "__main__":
    # Example usage:
    dataset = PlaceholderDataset(num_images=100)

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

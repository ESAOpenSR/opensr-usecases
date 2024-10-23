import torch
from torch.utils.data import Dataset
import numpy as np
import random
import matplotlib.pyplot as plt

class PlaceholderDataset(Dataset):
    def __init__(self, num_images=100,phase="test", image_type="lr"):
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

        # Generate 5 randomly sized and placed squares
        for _ in range(5):
            square_size = random.randint(10, 50)  # Random size for the square
            x = random.randint(0, self.width - square_size)
            y = random.randint(0, self.height - square_size)

            # Assign random color for the square (one for each band)
            color = np.random.randint(0, 256, (4,), dtype=np.uint8)

            # Place the square on the image
            image[y:y+square_size, x:x+square_size] = color

            # Update the binary mask where the square is
            mask[y:y+square_size, x:x+square_size] = 1

        # Convert the image to a PyTorch tensor and normalize to [0, 1] range
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Convert the mask to a PyTorch tensor
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

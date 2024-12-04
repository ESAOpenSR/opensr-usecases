import matplotlib.pyplot as plt
import os
import torch


def save_images(image_dict, save_dir="results/example_images"):
    """
    Generates and saves visual comparisons of input images, ground truth masks, 
    and predicted masks for high-resolution (HR), low-resolution (LR), and 
    super-resolution (SR) images.

    Parameters:
    - image_dict (dict): A dictionary containing the following keys:
        - "HR" (dict): Data related to high-resolution (HR) images.
            - "image" (list of torch.Tensor): Input HR images.
            - "GT" (list of torch.Tensor): Ground truth masks.
            - "pred" (list of torch.Tensor): Predicted masks for HR images.
        - "LR" (dict): Data related to low-resolution (LR) images.
            - "image" (list of torch.Tensor): Input LR images.
            - "pred" (list of torch.Tensor): Predicted masks for LR images.
        - "SR" (dict): Data related to super-resolution (SR) images.
            - "image" (list of torch.Tensor): Input SR images.
            - "pred" (list of torch.Tensor): Predicted masks for SR images.
    - save_dir (str, optional): The directory where the output images will be saved.
      Default is "results/example_images".

    Returns:
    None

    Saves:
    - For each example in `image_dict`, a 3x3 grid visualization is created and saved
      as an image file in `save_dir`.
        - Columns:
            1. Input Image
            2. Ground Truth Mask (GT)
            3. Predicted Mask
        - Rows:
            1. HR (High Resolution)
            2. LR (Low Resolution)
            3. SR (Super Resolution)

    Example:
    --------
    image_dict = {
        "HR": {"image": [...], "GT": [...], "pred": [...]},
        "LR": {"image": [...], "pred": [...]},
        "SR": {"image": [...], "pred": [...]}
    }
    save_images(image_dict, save_dir="results/example_images")
    
    This will save individual image comparison plots for each example in the specified directory.
    """
    
    
    for type in ["HR", "SR", "LR"]:
        assert type in image_dict.keys(), f"Key {type} not found in image_dict. Need to run analysis with image saving enabled."
    
    # extract images
    mask_hr = image_dict["HR"]["GT"]
    mask_lr = image_dict["LR"]["GT"]
    mask_sr = image_dict["SR"]["GT"]
    images_sr = image_dict["SR"]["image"]
    images_lr = image_dict["LR"]["image"]
    images_hr = image_dict["HR"]["image"]
    preds_sr = image_dict["SR"]["pred"]
    preds_lr = image_dict["LR"]["pred"]
    preds_hr = image_dict["HR"]["pred"]

    num_examples = len(images_sr)
    
    # Ensure the output directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Plot for each example
    for i in range(num_examples):
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        fig.suptitle(f"Performance Comparison", fontsize=16)
        plt.tight_layout(pad=2.0)

        # Titles for columns
        column_titles = ["Input Image", "Mask (GT)", "Predicted Mask"]
        for col_idx, title in enumerate(column_titles):
            axes[0, col_idx].set_title(title, fontsize=14)

        # Row titles
        row_titles = ["HR", "LR", "SR"]
        for row_idx, row_title in enumerate(row_titles):
            axes[row_idx, 0].annotate(row_title, xy=(-0.5, 0.5), xycoords="axes fraction",
                                      fontsize=14, ha="right", va="center", rotation=0)

        # Row 1: HR
        axes[0, 0].imshow(images_hr[i].permute(1, 2, 0)[:,:,:3].cpu().numpy())
        axes[0, 0].axis("off")
        axes[0, 1].imshow(mask_hr[i].cpu().numpy()[0], cmap="gray")
        axes[0, 1].axis("off")
        preds_hr_ = torch.where(preds_lr[i] > 0.6, torch.tensor(0.9999), torch.tensor(0.0))
        preds_hr_ = preds_hr_.permute(1, 2, 0).cpu().numpy()
        axes[0, 2].imshow(preds_hr_[0], cmap="gray")
        axes[0, 2].axis("off")

        # Row 2: LR
        axes[1, 0].imshow(images_lr[i].permute(1, 2, 0)[:,:,:3].cpu().numpy())
        axes[1, 0].axis("off")
        axes[1, 1].imshow(mask_lr[i].cpu().numpy()[0], cmap="gray")
        axes[1, 1].axis("off")
        preds_lr_ = torch.where(preds_lr[i] > 0.6, torch.tensor(0.9999), torch.tensor(0.0))
        preds_lr_ = preds_lr_.permute(1, 2, 0).cpu().numpy()
        axes[1, 2].imshow(preds_lr_[0], cmap="gray")
        axes[1, 2].axis("off")

        # Row 3: SR
        axes[2, 0].imshow(images_sr[i].permute(1, 2, 0)[:,:,:3].cpu().numpy())
        axes[2, 0].axis("off")
        axes[2, 1].imshow(mask_sr[i].cpu().numpy()[0], cmap="gray")
        axes[2, 1].axis("off")
        preds_sr_ = torch.where(preds_lr[i] > 0.6, torch.tensor(0.9999), torch.tensor(0.0))
        preds_sr_ = preds_sr_.permute(1, 2, 0).cpu().numpy()
        axes[2, 2].imshow(preds_sr_[0], cmap="gray")
        axes[2, 2].axis("off")

        # Adjust layout for tightness
        plt.subplots_adjust(wspace=0.05, hspace=0.2)

        # Save the figure
        save_path = os.path.join(save_dir, f"example_{i + 1}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")
     

def create_images(dataloader,model,device="cpu",num_images=10):
    """
    Processes a PyTorch DataLoader to generate images, ground truth targets, 
    and model predictions. It is useful for visualizing and analyzing model outputs.

    Parameters:
    - dataloader (torch.utils.data.DataLoader): DataLoader providing batches of images and targets.
    - model (torch.nn.Module): The trained PyTorch model used for generating predictions.
    - device (str, optional): The device to use for inference (default: "cpu").
    - num_images (int, optional): The maximum number of images to process (default: 10).

    Returns:
    dict: A dictionary containing:
        - "image" (list of torch.Tensor): Processed input images.
        - "GT" (list of torch.Tensor): Ground truth targets corresponding to the input images.
        - "pred" (list of torch.Tensor): Model predictions for the input images.
    """
    
    images,targets,outputs = [],[],[]
    
    c=0
    for im,tgt in dataloader:
        for im_,tgt_ in zip(im,tgt):
            c=c+1
            images.append(im_)
            targets.append(tgt_)
            with torch.no_grad():
                outputs.append(model(im_.unsqueeze(0).to(device)).squeeze(0).detach().cpu())
            if c>num_images:
                break
        if c>num_images:
            break
    return({"image":images,"GT":targets,"pred":outputs})

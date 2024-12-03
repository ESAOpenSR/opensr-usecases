import matplotlib.pyplot as plt
import os


def save_images(image_dict, save_dir="results/example_images"):
    for type in ["HR", "SR", "LR"]:
        assert type in image_dict.keys(), f"Key {type} not found in image_dict. Need to run analysis with image saving enabled."
    
    gts = image_dict["HR"]["GT"]
    images_sr = image_dict["SR"]["image"]
    images_lr = image_dict["LR"]["image"]
    images_hr = image_dict["HR"]["image"]
    preds_sr = image_dict["SR"]["pred"]
    preds_lr = image_dict["LR"]["pred"]
    preds_hr = image_dict["HR"]["pred"]

    num_examples = len(gts)
    
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
        axes[0, 0].imshow(images_hr[i].permute(1, 2, 0).cpu().numpy())
        axes[0, 0].axis("off")
        axes[0, 1].imshow(gts[i].cpu().numpy()[0], cmap="gray")
        axes[0, 1].axis("off")
        axes[0, 2].imshow(preds_hr[i].detach().cpu().numpy()[0], cmap="gray")
        axes[0, 2].axis("off")

        # Row 2: LR
        axes[1, 0].imshow(images_lr[i].permute(1, 2, 0).cpu().numpy())
        axes[1, 0].axis("off")
        axes[1, 1].imshow(gts[i].cpu().numpy()[0], cmap="gray")
        axes[1, 1].axis("off")
        axes[1, 2].imshow(preds_lr[i].detach().cpu().numpy()[0], cmap="gray")
        axes[1, 2].axis("off")

        # Row 3: SR
        axes[2, 0].imshow(images_sr[i].permute(1, 2, 0).cpu().numpy())
        axes[2, 0].axis("off")
        axes[2, 1].imshow(gts[i].cpu().numpy()[0], cmap="gray")
        axes[2, 1].axis("off")
        axes[2, 2].imshow(preds_sr[i].detach().cpu().numpy()[0], cmap="gray")
        axes[2, 2].axis("off")

        # Adjust layout for tightness
        plt.subplots_adjust(wspace=0.05, hspace=0.2)

        # Save the figure
        save_path = os.path.join(save_dir, f"example_{i + 1}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")
     

def create_images(dataloader,model,device="cpu"):
    images,targets,outputs = [],[],[]
    
    num_images = 10
    c=0
    for im,tgt in dataloader:
        for im_,tgt_ in zip(im,tgt):
            c=c+1
            images.append(im_)
            targets.append(tgt_)
            outputs.append(model(im_.unsqueeze(0).to(device)).squeeze(0))
    return({"image":images,"GT":targets,"pred":outputs})

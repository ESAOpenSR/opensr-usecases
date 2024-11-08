# Super-Resolution Model Validator for Segmentation Tasks

Welcome to the **Super-Resolution Model Validator** project! This tool is designed to help researchers and developers validate their **super-resolution models** in the context of **segmentation tasks**. The primary goal is to compare segmentation performance across **Low Resolution (LR)**, **High Resolution (HR)**, and **Super-Resolution (SR)** images using standard metrics as well as custom metrics that focus on object identification.

## Project Overview

In this project, users can submit their own dataset and models to evaluate how well the models perform **object segmentation** across different resolutions of images. This tool calculates a range of segmentation metrics and averages them over the dataset, providing insights into how the **resolution** of the input images (LR, HR, SR) affects the ability of the models to correctly segment objects.

The main focus of the validation process is to understand how well objects (e.g., buildings, in the case of remote sensing) are identified, and how this identification accuracy changes based on the input data type (LR, HR, or SR).

## Features

- **Customizable Dataset and Models**: Easily plug in your own dataset and models.
- **Multi-Resolution Comparison**: Validate models on LR, HR, and SR versions of images.
- **Standard Segmentation Metrics**: Computes metrics like IoU, Dice coefficient, Precision, Recall, and Accuracy.
- **Object Identification Metrics**: Special metrics that compute the percentage of objects correctly identified, focusing on size-based object identification.
- **Averaged Metrics**: Metrics are calculated for each batch and averaged across the entire dataset.
- **Debugging Support**: An optional debugging mode is available to limit the number of iterations for faster testing.
- **mAP Plotting**: Show how the mAP for object detecion looks like for each input data type.


## How It Works

### Input

- **Dataset**: The user provides a dataset containing images and ground truth segmentation masks.
- **Models**: The user provides models that perform segmentation tasks on LR, HR, and SR versions of the images. These models can be any pre-trained or custom segmentation models that output predicted masks.
  
### Metrics

The tool calculates the following metrics for each resolution (LR, HR, SR):

- **Intersection over Union (IoU)**: Measures the overlap between the predicted and ground truth masks.
- **Dice Coefficient**: Measures how well the predicted mask matches the ground truth.
- **Precision and Recall**: Standard metrics to evaluate the true positive rate and false positive rate for segmentation tasks.
- **Accuracy**: Measures the overall correct predictions in the segmentation task.

In addition, the tool computes **custom object identification metrics**:

- **Object Identification Percentage**: The percentage of objects that are correctly identified based on a given confidence threshold.
- **Size-Based Identification**: Metrics showing how well objects are identified based on their size (e.g., small vs. large objects).
  
### Output

The output of the tool is a set of averaged metrics for each resolution (LR, HR, SR). These results allow users to compare how well objects are segmented in different resolutions and understand how the use of **super-resolution** models impacts segmentation performance.

## Key Use Cases

1. **Super-Resolution Model Validation**: Assess how well your SR models improve segmentation tasks compared to LR and HR models.
2. **Segmentation Performance Analysis**: Analyze standard segmentation metrics alongside object-based metrics that track the percentage of correctly identified objects, especially for differently sized objects (e.g., small vs. large buildings).
3. **Model Comparison**: Compare segmentation performance across different models and resolutions to identify strengths and weaknesses.

## Getting Started

### Requirements

- Python 3.7+
- PyTorch
- tqdm (for progress bars)

### Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/your-repository/sr-segmentation-validator.git
    cd sr-segmentation-validator
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

To use this tool, you will need to follow these steps:

1. **Prepare Your Dataset**: Ensure that your dataset includes the images and ground truth segmentation masks.
2. **Define Your Models**: Provide models for LR, HR, and SR image segmentation. Each model should be capable of outputting a predicted mask for the input images.
3. **Run the Validation**: Use the provided `Validator` class to run the validation process and compute the metrics.

#### Example Code 

```python
from opensr_usecases.models.placeholder_model import PlaceholderModel
from opensr_usecases.data.placeholder_dataset import PlaceholderDataset

# Initialize the datasets - For LR,SR,HR
dataset_lr = PlaceholderDataset(phase="test", image_type="lr")
dataset_hr = PlaceholderDataset(phase="test", image_type="hr")
dataset_sr = PlaceholderDataset(phase="test", image_type="sr")

# Initialize dataloaders for each dataset
dataloader_lr = DataLoader(dataset_lr, batch_size=12, shuffle=False)
dataloader_hr = DataLoader(dataset_hr, batch_size=12, shuffle=False)
dataloader_sr = DataLoader(dataset_sr, batch_size=12, shuffle=False)

# Initialize different models for LR, HR, and SR
lr_model = PlaceholderModel()
hr_model = PlaceholderModel()
sr_model = PlaceholderModel()

# Create a Validator object
validator = Validator(device="cuda", debugging=True)

# Run validation for different resolutions
validator.calculate_masks_metrics(dataloader=dataloader_lr, model=lr_model, pred_type="LR", debugging=True)
validator.calculate_masks_metrics(dataloader=dataloader_hr, model=hr_model, pred_type="HR", debugging=True)
validator.calculate_masks_metrics(dataloader=dataloader_sr, model=sr_model, pred_type="SR", debugging=True)

# Retrieve and print the raw metrics
metrics = validator.return_raw_metrics()
validator.print_sr_improvement()


# calculate mAP curves
val_obj.get_mAP_curve(dataloader_lr, lr_model, pred_type="LR", amount_batches=10)
val_obj.get_mAP_curve(dataloader_hr, hr_model, pred_type="HR", amount_batches=10)
val_obj.get_mAP_curve(dataloader_sr, sr_model, pred_type="SR", amount_batches=10)

# plot mAP curve
mAP_plot = val_obj.plot_mAP_curve()
mAP_plot.save("resources/mAP_plot.png")

```

4. **Debugging**
If you want to quickly test or debug your models without running through the entire dataset, set the debugging flag to True. This will limit the evaluation to 10 batches:  
```python
validator = Validator(device="cuda", debugging=True)
```

## Main Functions  
- **calculate_masks_metrics(dataloader, model, pred_type)**: Predicts masks using the provided model and computes relevant segmentation metrics.
- **return_raw_metrics()**: Returns the raw metrics stored in the object.
- **print_raw_metrics()**: Prints the raw metrics stored in the object.
- **print_sr_improvement()**: Prints a table showing SR metric improvement over LR and loss over HR.
- **get_mAP_curve()**: Computes and stores the mean Average Precision (mAP) curve over multiple thresholds for a given model and dataset across a specified number of batches.
- **plot_mAP_curve()**: Plots the mAP curve for the stored prediction types and returns it as a PIL.Image.

## Example Output
### Impriovement Statistics
The tool generates a table comparing SR metric improvement over LR and loss over HR. Here's an example:
```sql
+----------------------------------+---------------------------+---------------------------+
|              Metric              | Improvement of SR over LR | Improvement of HR over SR |
+----------------------------------+---------------------------+---------------------------+
|          avg_obj_score           |          -0.0002          |          -0.0003          |
|          perc_found_obj          |          -3.8546          |           0.4100          |
| avg_obj_pred_score_by_size_11-20 |           0.0123          |          -0.0003          |
| avg_obj_pred_score_by_size_5-10  |          -0.0082          |           0.0071          |
|  avg_obj_pred_score_by_size_21+  |          -0.0032          |           0.0005          |
|  avg_obj_pred_score_by_size_0-4  |          -0.0571          |          -0.0001          |
|               IoU                |          -0.0001          |          -0.0000          |
|               Dice               |          -0.0002          |          -0.0001          |
|            Precision             |          -0.0001          |          -0.0001          |
|              Recall              |          -0.0004          |           0.0078          |
|             Accuracy             |          -0.0002          |          -0.0003          |
+----------------------------------+---------------------------+---------------------------+
```
### mAP Curve for Detected Objects
![mAP Curve](resources/mAP_plot.png?raw=true)

## Results and Analysis
At the end of the validation process, you will receive a set of metrics that show how well objects were identified and segmented across different resolutions. The results will include insights into how smaller and larger objects are affected by the resolution of the input images, allowing you to understand the performance trade-offs of using super-resolution models. If required, you will also see a mAP curve for each data type prediciton.

## Conclusion
The Super-Resolution Segmentation Validator provides a simple and effective way to validate your segmentation models across different image resolutions (LR, HR, SR). Use it to analyze and improve your models, while gaining insights into how resolution impacts segmentation performance.  
By comparing the results across LR, HR, and SR images, you can make informed decisions about the effectiveness of your super-resolution models and understand how resolution impacts segmentation tasks in your specific domain.


# Super-Resolution Model Validator for Segmentation Tasks

Welcome to the **Super-Resolution Model Validator** project! This tool is designed to help researchers and developers validate their **super-resolution models** in the context of **segmentation tasks**. The primary goal is to compare segmentation performance across **Low Resolution (LR)**, **High Resolution (HR)**, and **Super-Resolution (SR)** images using standard metrics as well as custom metrics that focus on object identification.

## Project Overview

In this project, users can submit their own dataset and models to evaluate how well the models perform object segmentation across different resolutions of images. This tool calculates a range of segmentation metrics and averages them over the dataset, providing insights into how the resolution of the input images (LR, HR, SR) affects the ability of the models to correctly segment objects.

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
- **Average Object Prediction Score**: An average of pixel predictions for each object
  
### Output

The output of the tool is a set of averaged metrics for each resolution (LR, HR, SR). These results allow users to compare how well objects are segmented in different resolutions and understand how the use of super-resolution models impacts segmentation performance. Cached results are automatically reused if force_recalc=False to speed up analysis.

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

#### Option A: Cloning
1. Clone this repository:

    ```bash
    https://github.com/ESAOpenSR/opensr-usecases
    cd opensr-usecases
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```
#### Option B: PIP
    ```bash
    pip install opensr-usecases
    ```


### Usage

To use this tool, you will need to follow these steps:

1. **Prepare Your Dataset**: Ensure that your dataset includes the images and ground truth segmentation masks.
2. **Define Your Models**: Provide models for LR, HR, and SR image segmentation. Each model should be capable of outputting a predicted mask for the input images.
3. **Run the Validation**: Use the provided `Validator` class to run the validation process and compute the metrics.
  
It is advisable that you follow the steps outlined in the demo.py file.

#### Example Code 

```python
# 0. Imports ----------------------------------------------------------------------------------------
from torch.utils.data import DataLoader
from tqdm import tqdm
from opensr_usecases import Validator


# 1. Get Data
# 1.1 Get Datasets
from opensr_usecases.data.placeholder_dataset import PlaceholderDataset
dataset_lr = PlaceholderDataset(phase="test", image_type="lr")
dataset_hr = PlaceholderDataset(phase="test", image_type="hr")
dataset_sr = PlaceholderDataset(phase="test", image_type="sr")

# 1.2 Create DataLoaders
dataloader_lr = DataLoader(dataset_lr, batch_size=4, shuffle=True)
dataloader_hr = DataLoader(dataset_hr, batch_size=4, shuffle=True)
dataloader_sr = DataLoader(dataset_sr, batch_size=4, shuffle=True)


# 2. Get Models -----------------------------------------------------------------------------------------------------
from opensr_usecases.models.placeholder_model import PlaceholderModel
lr_model = PlaceholderModel()
hr_model = PlaceholderModel()
sr_model = PlaceholderModel()


# 3. Validate -----------------------------------------------------------------------------------------------------
# 3.1 Create Validator object
val_obj = Validator(output_folder="data_folder", device="cpu", force_recalc= False, debugging=True)

# 3.2  Calculate images and save to Disk
val_obj.run_predictions(dataloader_lr, lr_model, pred_type="LR", load_pkl=True)
val_obj.run_predictions(dataloader_hr, hr_model, pred_type="HR", load_pkl=True)
val_obj.run_predictions(dataloader_sr, sr_model, pred_type="SR", load_pkl=True)

# 3.3 - Calcuate Metrics
# 3.3.1 Calculate Segmentation Metrics based on predictions
val_obj.calculate_segmentation_metrics(pred_type="LR", threshold=0.75)
val_obj.calculate_segmentation_metrics(pred_type="HR", threshold=0.75)
val_obj.calculate_segmentation_metrics(pred_type="SR", threshold=0.75)
    
# 3.3.2 Calculate Object Detection Metrics based on predictions
val_obj.calculate_object_detection_metrics(pred_type="LR", threshold=0.50)
val_obj.calculate_object_detection_metrics(pred_type="HR", threshold=0.50)
val_obj.calculate_object_detection_metrics(pred_type="SR", threshold=0.50)


# 4. Check out Results and Metrics -------------------------------------------------------------------------------------
# 4.1 Visual Inspection
val_obj.save_results_examples(num_examples=1)

# 4.2 Check Segmentation Metrics
val_obj.print_segmentation_metrics(save_csv=True)
val_obj.print_segmentation_improvements(save_csv=True)

# 4.3 Check Object Detection Metrics
val_obj.print_object_detection_metrics(save_csv=True)
val_obj.print_object_detection_improvements(save_csv=True)

# 4.4 Check Threshold Curves
val_obj.plot_threshold_curves(metric="all")
```

4. **Debugging**
If you want to quickly test or debug your models without running through the entire dataset, set the debugging flag to True. This will limit the evaluation to 10 batches:  
```python
validator = Validator(device="cuda", debugging=True)
```

## Main Functions  
- `run_predictions()`: Predicts and stores masks. Uses cached `.pkl` if available.
- `calculate_segmentation_metrics()`: Computes segmentation metrics. Uses cached values if available unless `force_recalc=True`.
- `calculate_object_detection_metrics()`: Computes object-level detection metrics.
- `plot_threshold_curves()`: Plots metrics vs thresholds. Loads `.pkl` if cached.
- `save_results_examples()`: Saves visual triplets of input, prediction, and GT for LR/SR/HR.
- `print_segmentation_metrics()`: Prints and optionally saves segmentation scores.
- `print_segmentation_improvements()`: Shows delta between SR and LR/HR.
- `print_object_detection_metrics()`: Prints and saves object detection metrics.
- `print_object_detection_improvements()`: Shows object-level detection improvements.


## Example Output
### Impriovement Statistics
The tool generates a table comparing SR metric improvement over LR and loss over HR. Here's an example:
```sql
+-----------+-----------+----------+-----------+
|   Metric  | LR → SR Δ |    SR    | HR → SR Δ |
+-----------+-----------+----------+-----------+
|    IoU    |    0.0    |   0.0    |    0.0    |
|    Dice   |    0.0    |   0.0    |    0.0    |
| Precision |    0.0    |   0.0    |    0.0    |
|   Recall  |    0.0    |   0.0    |    0.0    |
|  Accuracy |  0.004232 | 0.750163 |  0.007812 |
+-----------+-----------+----------+-----------+
```
  
```sql
+---------------------------------+-----------+----------+-----------+
|              Metric             | LR → SR Δ |    SR    | HR → SR Δ |
+---------------------------------+-----------+----------+-----------+
| Average Object Prediction Score |  0.004536 | 0.496032 |  0.004596 |
|    Percent of Buildings Found   |    10.0   |   40.0   | 14.583333 |
+---------------------------------+-----------+----------+-----------+
```

### mAP Curve for Detected Objects
![mAP Curve](resources/threshold_plot.png)

### mAP Curve for Detected Objects
![example images](resources/example.png)

## Results and Analysis
At the end of the validation process, you will receive a set of metrics that show how well objects were identified and segmented across different resolutions. The results will include insights into how smaller and larger objects are affected by the resolution of the input images, allowing you to understand the performance trade-offs of using super-resolution models. If required, you will also see a mAP curve for each data type prediciton.

## Conclusion
The Super-Resolution Segmentation Validator provides a simple and effective way to validate your segmentation models across different image resolutions (LR, HR, SR). Use it to analyze and improve your models, while gaining insights into how resolution impacts segmentation performance.  
By comparing the results across LR, HR, and SR images, you can make informed decisions about the effectiveness of your super-resolution models and understand how resolution impacts segmentation tasks in your specific domain.


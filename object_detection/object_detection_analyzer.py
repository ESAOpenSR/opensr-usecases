import numpy as np
from collections import defaultdict
import torch



class ObjectDetectionAnalyzer:
    def __init__(self):
        from object_detection.object_detection_utils import compute_avg_object_prediction_score
        from object_detection.object_detection_utils import compute_found_objects_percentage
        from object_detection.object_detection_utils import compute_avg_object_prediction_score_by_size
        from object_detection.object_detection_utils import standard_metrics
        self.compute_avg_object_prediction_score = compute_avg_object_prediction_score
        self.compute_found_objects_percentage = compute_found_objects_percentage
        self.compute_avg_object_prediction_score_by_size = compute_avg_object_prediction_score_by_size
        self.standard_metrics = standard_metrics
        
    def check_mask_validity(self,mask):
        if type(mask) == torch.Tensor:
            mask = mask.detach().cpu()
        if not type(mask) == np.ndarray:
            mask = mask.numpy()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        mask = mask.clip(0,1)
        return mask
        
    def compute(self,target, pred):
        # prepare masks
        target, pred = self.check_mask_validity(target), self.check_mask_validity(pred)
        
        # calculate Obj metrics
        metrics_dict = {    "avg_obj_score":self.compute_avg_object_prediction_score(target, pred),
                            "perc_found_obj":self.compute_found_objects_percentage(target, pred),
                            "avg_obj_pred_score_by_size":self.compute_avg_object_prediction_score_by_size(target, pred)
                         }
        # calculate and add standard Metrics to dict
        metrics_dict.update(self.standard_metrics(target, pred))
        
        # return
        return metrics_dict
    

if __name__ == "__main__":
    analyzer = ObjectDetectionAnalyzer()

    # Get some data
    from data.dataset_usa_buildings import SegmentationDataset
    ds = SegmentationDataset(phase="val",image_type="sr")
    im,mask = ds.__getitem__(10)

    # test
    metrics = analyzer.compute(target=mask, pred=np.random.rand(*mask.shape))





        
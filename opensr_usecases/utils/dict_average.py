def compute_average_metrics(metrics_list):
    """
    Computes the average of each metric from a list of dictionaries, handling nested dictionaries recursively.

    Args:
        metrics_list (list): A list of dictionaries, where each dictionary holds various metrics, 
                             including nested dictionaries.

    Returns:
        dict: A dictionary containing the average value of each metric.
    """
    
    def recursive_average(dict_list):
        """
        A recursive helper function to compute averages for nested dictionaries.
        
        Args:
            dict_list (list): A list of dictionaries.
            
        Returns:
            dict: A dictionary with averaged values.
        """
        average_metrics = {}

        for metrics in dict_list:
            for key, value in metrics.items():
                # If the value is a dictionary, call recursively
                if isinstance(value, dict):
                    if key not in average_metrics:
                        average_metrics[key] = []
                    average_metrics[key].append(value)
                else:
                    if key in average_metrics:
                        average_metrics[key] += value
                    else:
                        average_metrics[key] = value
        
        # Compute the average for each metric, handling nested dictionaries recursively
        num_entries = len(dict_list)
        for key in average_metrics:
            if isinstance(average_metrics[key], list) and isinstance(average_metrics[key][0], dict):
                # Recursive case: compute averages for nested dictionaries
                average_metrics[key] = recursive_average(average_metrics[key])
            else:
                average_metrics[key] /= num_entries
        
        return average_metrics
    
    return recursive_average(metrics_list)
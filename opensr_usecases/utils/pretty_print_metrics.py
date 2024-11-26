from prettytable import PrettyTable
import os

def flatten_metrics(metrics, parent_key='', sep='_'):
    """
    Recursively flattens a nested dictionary for easy comparison in tables.
    
    Args:
        metrics (dict): The nested dictionary to flatten.
        parent_key (str): The prefix for keys in the flattened dictionary.
        sep (str): Separator used to concatenate parent and child keys.
    
    Returns:
        dict: A flattened dictionary.
    """
    items = []
    for k, v in metrics.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_metrics(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def print_sr_improvement(metrics,save_to_txt=False):
    """
    Prints a table showing SR metric "improvement" over LR and "loss" over HR.
    
    Args:
        metrics (dict): A dictionary containing LR, HR, and SR metrics.
    """
    
    if 'SR' not in metrics or 'LR' not in metrics or 'HR' not in metrics:
        print("Metrics for SR, LR, and HR are required.")
        return

    sr_metrics = flatten_metrics(metrics['SR'])
    lr_metrics = flatten_metrics(metrics['LR'])
    hr_metrics = flatten_metrics(metrics['HR'])

    # Define the table
    table = PrettyTable()
    table.field_names = ["Metric", "Improvement of SR over LR","Improvement of HR over SR"]
    
    # Compare metrics between SR, LR, and HR
    for metric in sr_metrics:
        if isinstance(sr_metrics[metric], (float, int)):  # If it's a numeric metric, compare
            improvement = sr_metrics[metric] - lr_metrics.get(metric, 0)
            loss = sr_metrics[metric] - hr_metrics.get(metric, 0)
            table.add_row([metric, f"{improvement:.4f}", f"{loss:.4f}"])
    
    # Print the table, save if wanted
    print(table)
    
    # create folder if necessary
    if not os.path.exists("results"):
        os.makedirs("results")
    if save_to_txt!=False:
        with open('results/tabular_results.txt', 'w') as f:
            f.write(str(table))


if __name__ == "__main__":
    # Example usage with nested metrics
    metrics = {
        'LR': {'avg_obj_score': 0.5025333243978853, 'perc_found_obj': 53.42380360431844, 
            'avg_obj_pred_score_by_size': {'11-20': 0.5008569441493559, '5-10': 0.392170871069593, '21+': 0.5004807137816203, '0-4': 0.40658042572363456}, 
            'IoU': 0.021808707095858835, 'Dice': 0.0401833480704261},
        'SR': {'avg_obj_score': 0.6025333243978853, 'perc_found_obj': 60.42380360431844, 
            'avg_obj_pred_score_by_size': {'11-20': 0.6008569441493559, '5-10': 0.492170871069593, '21+': 0.6004807137816203, '0-4': 0.50658042572363456}, 
            'IoU': 0.12180870709585884, 'Dice': 0.1401833480704261},
        'HR': {'avg_obj_score': 0.7025333243978853, 'perc_found_obj': 70.42380360431844, 
            'avg_obj_pred_score_by_size': {'11-20': 0.7008569441493559, '5-10': 0.592170871069593, '21+': 0.7004807137816203, '0-4': 0.60658042572363456}, 
            'IoU': 0.22180870709585884, 'Dice': 0.2401833480704261}
    }

    # Call the function
    print_sr_improvement(metrics)
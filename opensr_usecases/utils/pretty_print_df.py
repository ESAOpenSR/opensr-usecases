from prettytable import PrettyTable
import pandas as pd


def print_pretty_dataframe2(df, index_name="", float_round=4):
    """
    Pretty-print any pandas DataFrame using PrettyTable.

    Args:
        df (pd.DataFrame): The DataFrame to print.
        index_name (str): Label to use for the index column.
        float_round (int): Number of decimal places to round float values.
    """
    table = PrettyTable()

    # Set column headers
    table.field_names = [index_name] + list(df.columns)

    for idx, row in df.iterrows():
        values = [round(v, float_round) if isinstance(v, float) else v for v in row.values]
        table.add_row([idx] + values)

    print(table)
    
def print_pretty_dataframe(df, index_name="", float_round=4):
    """
    Pretty-print any pandas DataFrame using PrettyTable.
    Supports collapsing repeated index cells for nicer grouped formatting.

    Args:
        df (pd.DataFrame): The DataFrame to print.
        index_name (str): Label to use for the index column (or left blank).
        float_round (int): Decimal places to round float values.
    """
    import pandas as pd
    from prettytable import PrettyTable

    table = PrettyTable()

    # Handle single- or multi-index
    if isinstance(df.index, pd.MultiIndex):
        index_levels = df.index.names
        table.field_names = index_levels + list(df.columns)
        last_index = [None] * len(index_levels)

        for idx, row in df.iterrows():
            index_cells = []
            for level, val in zip(range(len(index_levels)), idx):
                if val == last_index[level]:
                    index_cells.append("")
                else:
                    index_cells.append(val)
                last_index[level] = val

            formatted_values = []
            for col, val in zip(df.columns, row):
                if col in {"Total", "True Positives", "False Negatives"} and pd.notna(val):
                    formatted_values.append(int(val))
                elif isinstance(val, float):
                    formatted_values.append(round(val, float_round))
                else:
                    formatted_values.append(val)

            table.add_row(index_cells + formatted_values)

    else:
        table.field_names = [index_name] + list(df.columns)
        last_index = None
        for idx, row in df.iterrows():
            index_cell = "" if idx == last_index else idx
            last_index = idx

            formatted_values = []
            for col, val in zip(df.columns, row):
                if col in {"Total", "True Positives", "False Negatives"} and pd.notna(val):
                    formatted_values.append(int(val))
                elif isinstance(val, float):
                    formatted_values.append(round(val, float_round))
                else:
                    formatted_values.append(val)

            table.add_row([index_cell] + formatted_values)

    print(table)

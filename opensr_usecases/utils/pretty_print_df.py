from prettytable import PrettyTable

def print_pretty_dataframe(df, index_name="", float_round=4, table_name=None):
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

    if table_name is not None:
        print(table_name)
    print(table)
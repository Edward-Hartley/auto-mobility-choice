import pandas as pd

trip_path_residents = 'data/Residents_TripData_KendallSquareCBGs_thursday_sep2019-nov2019.csv'
trip_path_workers = 'data/WorkersStudents_TripData_KendallSquareCBGs_thursday_sep2019-nov2019.csv'

def duplicate_values(data, column):
    """
    Returns the duplicate values in a column.
    """
    return data[column].value_counts()[data[column].value_counts() > 1]

def remove_duplicate_values(data, column):
    """
    Removes duplicate values in a column.
    """
    return data[~data[column].duplicated()]

def concat_columns(data, columns):
    """
    Concatenates the columns into a single column.
    """
    return data[columns].apply(lambda x: ' '.join(x), axis=1)

def unique_values(data, column):
    """
    Returns the unique values in a column.
    """
    return data[column].unique()

def concat_data(data1, data2):
    """
    Concatenates two dataframes.
    """
    return pd.concat([data1, data2])

def get_rows(data, column, value):
    """
    Returns the rows where the value is in the column.
    """
    return data[data[column] == value]

def get_rows_with_duplicate_values(data, column):
    """
    Returns the rows where the value is in the column.
    """
    return data[data[column].duplicated()]

def sort_data(data, column):
    """
    Sorts the dataframe by the column.
    """
    return data.sort_values(by=column)

def get_data(file_name):
    """
    Reads in a csv file and returns a pandas dataframe.
    """
    return pd.read_csv(file_name)

def print_data(data):
    """
    Prints the dataframe.
    """
    print(data)

def print_data_info(data):
    """
    Prints the dataframe info.
    """
    print(data.info())

def store_data(data, file_name):
    """
    Stores the dataframe as a csv file.
    """
    data.to_csv(file_name)

def combined_unique_rows(data1, data2, column):
    """
    Returns each unique row from the combined dataframe.
    """
    return remove_duplicate_values(concat_data(data1, data2), column)
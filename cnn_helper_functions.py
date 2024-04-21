import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#--------------------------------------------------------------------------------------#

def generate_dataset(path):
    """
    Generate a dataset of image file paths and labels.

    Parameters:
    - path (str): Path to directory with subdirectories of images.

    Returns:
    - DataFrame: Pandas DataFrame with 'imgpath' and 'labels' columns.
    """
    data = {'imgpath': [], 'labels': []}
    
    folders = os.listdir(path)
    
    for folder in folders:
        folderpath = os.path.join(path, folder)
        files = os.listdir(folderpath)
        
        for file in files:
            filepath = os.path.join(folderpath, file)
            
            data['imgpath'].append(filepath)
            data['labels'].append(folder)
    
    return pd.DataFrame(data)

#--------------------------------------------------------------------------------------#

def dataset_splitter(dataset, train_size = 0.9, test_size = 0.4, shuffle = True, random_state = 0):
    """
    Split a dataset into training, validation, and test sets.

    Parameters:
    - dataset (DataFrame): Pandas DataFrame containing the dataset.
    - train_size (float): Proportion of the dataset to include in the training set.
    - test_size (float): Proportion of the dataset to include in the test set.
    - shuffle (bool): Whether to shuffle the dataset before splitting.
    - random_state (int): Seed used by the random number generator for shuffling.

    Returns:
    - tuple: Three DataFrames (train_df, val_df, test_df) containing the training, validation, and test sets respectively.
    """
    train_df, temp_df = train_test_split(dataset, train_size = train_size, shuffle = shuffle, random_state = random_state)
    val_df, test_df = train_test_split(temp_df, test_size = test_size, shuffle = shuffle, random_state = random_state)

    train_df = train_df.reset_index(drop = True)
    val_df = val_df.reset_index(drop = True)
    test_df = test_df.reset_index(drop = True)
    
    return train_df, val_df, test_df

#--------------------------------------------------------------------------------------#
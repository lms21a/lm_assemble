from datetime import datetime
import time
from contextlib import contextmanager
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from typing import List


def timestamp():
    # Get the current date and time
    now = datetime.now()

    # Format the date and time
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")

    return formatted_now

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_memory_usage(model, input_size, dtype=torch.float32):
    """
    Estimates the memory usage of a PyTorch model in megabytes (MB).
    
    Args:
    model (torch.nn.Module): The PyTorch model.
    input_size (tuple): The size of the input tensor.
    dtype: The data type of the model parameters (default: torch.float32).

    Returns:
    float: Estimated memory usage in MB.
    """

    param_size = sum(p.numel() for p in model.parameters()) * torch.tensor(1, dtype=dtype).element_size()

    with torch.inference_mode():
        input_tensor = torch.zeros(input_size, dtype=dtype)
        output = model(input_tensor)
        output_size = output.numel() * torch.tensor(1, dtype=dtype).element_size()

    total_memory = (param_size + output_size) / (1024 ** 2)

    return total_memory

def tik():
    global START_TIME
    START_TIME = time.time()

def tok():
    end_time = time.time()
    if START_TIME is None:
        raise ValueError("tik() must be called before tok()")
    elapsed_time = end_time - START_TIME
    print(f"Elapsed time: {elapsed_time} seconds")
    return elapsed_time

@contextmanager
def timing(label = None):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"{label}: {end - start} seconds")

def save_checkpoint(state: dict, filename: str) -> None:
    checkpoint_dir = 'checkpoints'
    filename = os.path.join(checkpoint_dir, filename)
    torch.save(state, filename)

    return 0

def flatten_list(l: List[List]) -> List:
    return [j for k in l for j in k]

def load_checkpoint(model, optimizer, scaler, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scaler.load_state_dict(checkpoint['scaler_state'])
    
    return checkpoint['step'] 

def clear_folder_contents(folder_path, ignore_file_types=None):
    """
    Clear out all the contents of the specified folder without deleting the folder itself,
    with an option to ignore files of certain types.

    :param folder_path: Path to the folder whose contents are to be cleared.
    :param ignore_file_types: List of file extensions to ignore, or None to clear all files. Example: ['.npy', '.txt']
    """
    # Loop through each item in the folder
    for filename in tqdm(os.listdir(folder_path), desc='Clearing Out Folder'):
        file_path = os.path.join(folder_path, filename)

        # Check if the file type should be ignored
        if ignore_file_types and any(file_path.endswith(ext) for ext in ignore_file_types):
            continue

        try:
            # Remove the item
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # Remove file or link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directory
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def plot_csv(csv_filepath, accuracy=False):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_filepath)
    
    # Plotting Loss (Train and Test)
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(df['Step'], df['Train_Loss'], label='Train Loss')
    plt.plot(df['Step'], df['Test_Loss'], label='Test Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss over Steps')
    plt.legend()
    
    if accuracy:
        # Plotting Accuracy (Train and Test)
        plt.subplot(1, 2, 2)
        plt.plot(df['Step'], df['Train_Acc'], label='Train Accuracy')
        plt.plot(df['Step'], df['Test_Acc'], label='Test Accuracy')
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Steps')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

def plot_csvs(csv_filepaths, accuracy=False):
    plt.figure(figsize=(14, 6))
    
    for csv_filepath in csv_filepaths:
        # Read each CSV file into a DataFrame
        df = pd.read_csv(csv_filepath)
        
        # Assuming each CSV file has a unique identifier to distinguish them in the plot
        # You can adjust the unique_identifier logic as needed
        unique_identifier = csv_filepath.split('/')[-1].split('.')[0]  # Extract filename without extension
        
        # Plotting Loss (Train and Test)
        plt.subplot(1, 2, 1)
        plt.plot(df['Step'], df['Train_Loss'], label=f'Train Loss ({unique_identifier})')
        plt.plot(df['Step'], df['Test_Loss'], label=f'Test Loss ({unique_identifier})')
    
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss over Steps')
    plt.legend()
    
    if accuracy:
        plt.subplot(1, 2, 2)
        for csv_filepath in csv_filepaths:
            # Read each CSV file into a DataFrame
            df = pd.read_csv(csv_filepath)
            
            # Assuming each CSV file has a unique identifier to distinguish them in the plot
            unique_identifier = csv_filepath.split('/')[-1].split('.')[0]  # Extract filename without extension
            
            # Plotting Accuracy (Train and Test)
            plt.plot(df['Step'], df['Train_Acc'], label=f'Train Accuracy ({unique_identifier})')
            plt.plot(df['Step'], df['Test_Acc'], label=f'Test Accuracy ({unique_identifier})')
        
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Steps')
        plt.legend()
        
    plt.tight_layout()
    plt.show()
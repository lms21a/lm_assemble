import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from tokenizer import Tokenizer

import os
import shutil
from functools import partial

def tokenize(example, tokenizer):
    tokens = tokenizer.encode(example['captions'])
    return {'tokens': tokens}

def token_length(examples):
    num_tokens = [len(e) for e in examples['tokens']]
    return {'num_tokens': num_tokens}

def create_memmap(dataset, filename, dtype=np.int16):
    total_tokens = sum(dataset['num_tokens'])

    memmap_array = np.memmap(filename, dtype=dtype, mode='w+', shape=(total_tokens,))
    
    current_index = 0

    for example in dataset:
        tokens = example['tokens']
        num_tokens = example['num_tokens']

        memmap_array[current_index:current_index + num_tokens] = tokens
        current_index += num_tokens

    return memmap_array

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

if __name__ == '__main__':
    
    model_file = os.path.join('saved_models','spm_model.model')    
    test_size = .1

    tokenizer = Tokenizer(model_file=model_file)

    ds = load_dataset("RamAnanth1/lex-fridman-podcasts", cache_dir='data', split='train')

    tokenizer.train_tokenizer(
        dataset=ds.select_columns(['captions']),
        vocab_size=1024,
        tokenizer_sample_size=.1
    )

    tokenizer.load_tokenizer(add_bos=True, add_eos=True)

    tokenize_fn = partial(tokenize, tokenizer=tokenizer)
    rm_cols = list(ds.features)
    ds = ds.map(tokenize_fn, batched=True, remove_columns=rm_cols)
    
    ds = ds.shuffle().train_test_split(test_size=test_size)
    ds = ds.map(token_length, batched=True)

    train_memmap = create_memmap(ds['train'], 'data/train_memmap.tokens')
    test_memmap = create_memmap(ds['test'], 'data/test_memmap.tokens')

    train_memmap.flush()
    test_memmap.flush()

    clear_folder_contents('data', ignore_file_types=['.tokens'])
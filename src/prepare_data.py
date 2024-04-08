import os
import fire
import shutil
import numpy as np
from tqdm import tqdm
from functools import partial
from datasets import load_dataset
from datatools.tokenizer import Tokenizer

def tokenize(example, feature_col, tokenizer):
    tokens = tokenizer.encode(example[feature_col])
    num_tokens = [len(t) for t in tokens]
    return {'tokens': tokens, 'num_tokens': num_tokens}

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

def prep_shakespeare(vocab_size: int) -> None:

    dtype = np.int16 if vocab_size < 65536 else np.int32

    tokenizer = Tokenizer('saved_models/shakespeare_tokenizer.model')

    tokenizer.train_tokenizer(data = 'data/shakespeare.txt', vocab_size=vocab_size)

    with open('data/shakespeare.txt', 'r') as file:
        text = file.read()

    tokens = np.array(tokenizer.encode(text), dtype=dtype)
    np.save('data/shakespeare_tokens.npy', tokens)
    clear_folder_contents('data/', ignore_file_types=['.npy', '.tokens'])

def prep_lex(vocab_size: int, tokenizer_sample_size: float = .1):
    feature_col = 'captions'
    dtype = np.int16 if vocab_size < 65536 else np.int32

    ds = load_dataset("RamAnanth1/lex-fridman-podcasts", cache_dir='data', split='train')

    tokenizer = Tokenizer('saved_models/lex_tokenizer.model')
    
    tokenizer.train_tokenizer(
        data=ds.select_columns([feature_col]),
        vocab_size=vocab_size,
        tokenizer_sample_size=tokenizer_sample_size
    )

    tokenize_fn = partial(tokenize, feature_col=feature_col, tokenizer=tokenizer)
    rm_cols = list(ds.features)
    ds = ds.map(tokenize_fn, batched=True, remove_columns=rm_cols)

    data_memmap = create_memmap(ds, 'data/lex.tokens', dtype=dtype)
    data_memmap.flush()

    clear_folder_contents('data/', ignore_file_types=['.npy', '.tokens'])

def run_prep(dataset: str, vocab_size: int = 8000, tokenizer_sample_size: float = .1) -> None:
    if dataset == 'shakespeare':
        prep_shakespeare(vocab_size)
    elif dataset == 'lex':
        prep_lex(vocab_size, tokenizer_sample_size)

if __name__ == '__main__':
    fire.Fire(run_prep)
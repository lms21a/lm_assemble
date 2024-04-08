from datasets import load_dataset
import os
from functools import partial
from datatools.tokenizer import Tokenizer
from utils.helper_func import clear_folder_contents

def tokenize(example, feature_col, tokenizer):
    tokens = tokenizer.encode(example[feature_col])
    num_tokens = [len(t) for t in tokens]
    return {'tokens': tokens, 'num_tokens': num_tokens}

if __name__ == '__main__':

    ds = load_dataset("ag_news", split='train', cache_dir='data', data_dir='data')
    model_file = os.path.join('saved_models', 'agnews.model')
    tokenizer = Tokenizer(model_file)
    feature_col = 'text'
    
    tokenizer.train_tokenizer(dataset=ds.select_columns(['text']), vocab_size=512, tokenizer_sample_size=.005)
    tokenizer.load_tokenizer(add_bos=True, add_eos=True)
    tokenize_fn = partial(tokenize, tokenizer=tokenizer, feature_col = feature_col)


    ds = ds.map(tokenize_fn, batched=True)
    ds = ds.remove_columns(['text'])
    ds = ds.train_test_split(test_size=.1)
    ds.save_to_disk('data/ag_news-ds')
    clear_folder_contents('data', ignore_file_types=['.tokens', '-ds'])
import io
import os
import numpy as np
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from typing import List
from datasets import Dataset

class Tokenizer:

    def __init__(self, model_file: str, add_bos: bool = True, add_eos: bool = True):
        self.model_file = model_file
        self.add_bos = add_bos
        self.add_eos = add_eos

        self._load_model()
    
    def _load_model(self) -> None:
        if os.path.exists(self.model_file):
            self.model = SentencePieceProcessor(self.model_file, add_bos=self.add_bos, add_eos=self.add_eos)

    def encode(self, text: str | List[str]) -> List[int] | List[List[int]]:
        return self.model.Encode(text)

    def decode(self, tokens: List[int] | List[List[int]]) -> str | List[str]:
        return self.model.Decode(tokens)

    def _train_text(self, data: str, vocab_size: int, tokenizer_gen_kwargs: dict) -> None:
        SentencePieceTrainer.Train(
                input=data,
                vocab_size=vocab_size,
                max_sentence_length=1000000000,
                **tokenizer_gen_kwargs
            )

    def _train_dataset(
            self, data: Dataset, vocab_size: int, tokenizer_sample_size: float, tokenizer_gen_kwargs: dict
    ) -> None:
        
        assert data.num_columns == 1, 'Must Select Column to Tokenize'
        assert tokenizer_sample_size is not None, 'Must Provide Sample Size'
        
        samples = np.random.randint(0, data.num_rows-1, int(tokenizer_sample_size * data.num_rows))
        data_io = io.StringIO()
        
        idx = data.column_names[0]

        sentence_length = []
        for sample in samples:
            text = data[idx][sample] + '\n'
            sentence_length.append(len(text))
            data_io.write(text)
        
        data_io.seek(0)
        
        SentencePieceTrainer.train(
            sentence_iterator=data_io,
            vocab_size = vocab_size,
            max_sentence_length = max(sentence_length) + 1000,
            **tokenizer_gen_kwargs
        )

        data_io.flush()
        data_io.close()


    def train_tokenizer(
            self, data: str | Dataset, vocab_size: int, tokenizer_sample_size: int | None = None
    ) -> None:

        tokenizer_gen_kwargs = {
            'model_prefix': self.model_file.removesuffix('.model'),
            'model_type': 'bpe',
            'normalization_rule_name': 'identity',
            'shuffle_input_sentence': True,
            'character_coverage': .9995,
            'byte_fallback': True,
            'split_digits': True,
            'split_by_unicode_script': True,
            'split_by_whitespace': True,
            'split_by_number': True,
            'add_dummy_prefix': True
        }

        if isinstance(data, str):
            self._train_text(data, vocab_size, tokenizer_gen_kwargs)
            
        elif isinstance(data, Dataset):
            self._train_dataset(data, vocab_size, tokenizer_sample_size, tokenizer_gen_kwargs)

        else:
            raise NotImplementedError

        self._load_model()
import sentencepiece as spm
from typing import List, Union
import numpy as np
import io

class Tokenizer:
    def __init__(self, model_file: str, model_type: str = 'bpe') -> None:
        assert model_file.endswith('.model'), "Model File Must end with .model to match SentencePiece"
        self.model_file: str = model_file
        self.model_type: str = model_type
        
        self.sp: Union[None, spm.SentencePieceProcessor] = None

    def load_tokenizer(self, add_bos: bool = False, add_eos: bool = False) -> None:
        self.sp = spm.SentencePieceProcessor(model_file=self.model_file, add_bos=add_bos, add_eos=add_eos)

    def _ensure_loaded(self) -> None:
        assert self.sp is not None, "Tokenizer model is not loaded. Call load_tokenizer first."
    
    def _remove_suffix(self) -> None:
        if self.model_file.endswith('.model'):
            return self.model_file[:-6]

    def encode(self, text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        self._ensure_loaded()
        return self.sp.Encode(text)

    def decode(self, tokens: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        self._ensure_loaded()
        return self.sp.Decode(tokens)
    
    def train_tokenizer(self, dataset, vocab_size: int, tokenizer_sample_size: float) -> None:
        assert dataset.num_columns == 1, 'Must Select Column to Tokenize'
        
        samples = np.random.randint(0, dataset.num_rows-1, int(tokenizer_sample_size * dataset.num_rows))
        data_io = io.StringIO()
        
        idx = dataset.column_names[0]

        sentence_length = []
        for sample in samples:
            text = dataset[idx][sample] + '\n'
            sentence_length.append(len(text))
            data_io.write(text)
        
        data_io.seek(0)
        
        spm.SentencePieceTrainer.train(
            sentence_iterator=data_io,
            model_prefix = self._remove_suffix(),
            model_type = self.model_type,
            vocab_size = vocab_size,
            max_sentence_length = max(sentence_length) + 100
        )

        data_io.flush()
        data_io.close()
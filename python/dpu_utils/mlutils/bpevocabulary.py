import logging
import os
from collections import Counter
from tempfile import TemporaryDirectory
from typing import List, Optional, Sized, Union, Iterable
import sentencepiece as spm
import typing

__all__ = ['BpeVocabulary']

SPIECE_UNDERLINE = u'▁'


class BpeVocabulary(Sized):
    """
    A vocabulary that maps strings to unique ids (and back), and a tokenizer based on
    Byte-Pair Encoding using sentencepiece from https://github.com/google/sentencepiece.

        Sennrich, Rico, Barry Haddow, and Alexandra Birch.
        "Neural machine translation of rare words with subword units."
        arXiv preprint arXiv:1508.07909 (2015).

    To create a vocabulary use `BpeVocabulary.create_vocabulary()`.
    The control flow symbols have to be introduced
    manually, during preprocessing step.

    BpeVocabulary object usage: Assuming an initialized vocabulary `v`:

       * To get the the tokenized version of a string `v.tokenize("a single string here")`.
       * To get the ids of a string, use `v.get_id_or_unk_for_text("a single string here")`.
       * To get a string from a list of the ids of pieces, use `v.convert_ids_to_string([10, 2, 5, 3])`.
       * To get the size of the vocabulary use `len(v)`
    """
    LOGGER = logging.getLogger('BpeVocabulary')
    DEFAULT_CONTROL_SYMBOLS = ["<endofline>", "<endoftext>"]

    def __init__(self, max_size: int, sentencepiece_model_filepath: Optional[str]=None,
                 bos_token: str="<s>", eos_token: str="</s>", unk_token: str="<unk>", pad_token: str="<pad>",
                 user_defined_symbols: Optional[List[str]] = None,
                 control_symbols: Optional[List[str]]=None) -> None:

        self.__max_size=max_size
        self.__bos_token=bos_token
        self.__eos_token=eos_token
        self.__unk_token=unk_token
        self.__pad_token=pad_token

        self.vocab_file = sentencepiece_model_filepath
        if user_defined_symbols is None:
            user_defined_symbols = []
        self.user_defined_symbols=",".join(user_defined_symbols)

        if control_symbols is None:
            control_symbols = self.DEFAULT_CONTROL_SYMBOLS
        self.control_symbols=",".join(control_symbols)

        self.__sp_model = spm.SentencePieceProcessor()
        if sentencepiece_model_filepath is not None:
            self.__load_model_from_filepath(sentencepiece_model_filepath)

    #region Custom Pickling
    def __load_model_from_filepath(self, sentencepiece_model_filepath) -> bool:
        loaded = self.__sp_model.Load(sentencepiece_model_filepath)
        # We want to encapsulate all vocabulary-related elements in a single location (this object!) and avoid
        # dangling files. We store all model data in this object as a set of bytes.
        with open(sentencepiece_model_filepath, 'rb') as f:
            self.__sp_model_data = f.read()

        return loaded

    def __getstate__(self):
        """The __sp_model cannot be serialized. Remove it when pickling."""
        state = self.__dict__.copy()
        del state['_BpeVocabulary__sp_model']
        return state

    def __setstate__(self, state):
        """Restore __sp_model that could not be serialized."""
        self.__dict__.update(state)
        if self.__sp_model_data is None:
            return
        with TemporaryDirectory() as tmp_dir:
            model_file = os.path.join(tmp_dir, 'tmp.model')
            with open(model_file, 'wb') as f:
                f.write(self.__sp_model_data)
            self.__sp_model = spm.SentencePieceProcessor()
            self.__sp_model.Load(model_file)
    #endregion

    def get_pad(self) -> str:
        """ Get padding token. """
        if self.__pad_token is None:
            self.LOGGER.error("Using pad_token, but it is not set yet.")
        return self.__pad_token

    def get_unk(self) -> str:
        """ Get unknown token. """
        if self.__unk_token is None:
            self.LOGGER.error("Using unk_token, but it is not set yet.")
        return self.__unk_token

    def __len__(self) -> int:
        return len(self.__sp_model)

    def tokenize(self, text: str) -> List[str]:
        """ Tokenize a string. """
        pieces = self.__sp_model.EncodeAsPieces(text)

        new_pieces = []   # type: List[str]
        for piece in pieces:
            # Split subtokens composed of a digit and comma
            #
            # E.g. given in an input sentence: 
            #      text = 'for i in range(100, 2):'
            # Default output of tokenizer may be: 
            #      ['▁for', '▁i', '▁in', '▁range', '(1', '00,', '▁2', '):']
            # Following will change this to: 
            #      ['▁for', '▁i', '▁in', '▁range', '(1', '0', '0', ',', '▁2', '):']            
            if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
                cur_pieces = self.__sp_model.EncodeAsPieces(
                    piece[:-1].replace(SPIECE_UNDERLINE, ''))
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)

        return new_pieces

    def get_id_or_unk_for_text(self, text: str, pad_to_size: Optional[int] = None,
                               padding_element: int = 0) -> List[int]:
        """
        Tokenize (using BPE) a given string and return a list of the int ids of the wordpieces of the string.
        """
        tokens = self.tokenize(text)

        if pad_to_size is not None:
            tokens = tokens[:pad_to_size]

        ids = [self.__sp_model.PieceToId(t) for t in tokens]
        if pad_to_size is not None and len(ids) != pad_to_size:
            ids += [padding_element] * (pad_to_size - len(ids))

        return ids

    def convert_ids_to_string(self, piece_ids: List[int]) -> str:
        """Converts a sequence of piece ids (strings for sub-words) in a single string."""
        out_string = ''.join((self.__sp_model.IdToPiece(i) for i in piece_ids)).replace(SPIECE_UNDERLINE, ' ').strip()
        return out_string

    def create_vocabulary_from_file(self, sp_text_file: str, num_threads: Optional[int]=os.cpu_count(),
                                    max_sentence_length: int=16384, character_coverage: float=0.99995) -> None:
        """
        Train sentencepiece tokenizer using BPE model and build a vocabulary.

        sp_text_file: path to a plain text file containing the training dataset.
        """
        if num_threads is None:
            num_threads = 1
        with TemporaryDirectory() as tmpdir:
            model_filename = os.path.join(tmpdir, f'bpe_{self.__max_size}')
            command = [
                        f"--input={sp_text_file}",
                        f"--num_threads={num_threads}",
                        f"--model_prefix={model_filename}",
                        f"--vocab_size={self.__max_size}",
                        f"--model_type=bpe",
                        f"--max_sentence_length={max_sentence_length}",
                        f"--bos_piece={self.__bos_token}",
                        f"--eos_piece={self.__eos_token}",
                        f"--pad_piece={self.__pad_token}",
                        f"--pad_id=3",
                        f"--unk_piece={self.__unk_token}",
                        f"--user_defined_symbols={self.user_defined_symbols}",
                        f"--control_symbols={self.control_symbols}",
                        f"--character_coverage={character_coverage}",
                        "--minloglevel=1",
                        "--hard_vocab_limit=false",
                    ]

            spm.SentencePieceTrainer.train(
                " ".join(command)
            )

            loaded = self.__load_model_from_filepath(model_filename+'.model')
            assert loaded, 'Sentencepiece failed to load model.'

    def create_vocabulary(self, tokens: Union[Iterable[str], Iterable[List[str]], typing.Counter[str]]) -> None:
        with TemporaryDirectory() as dir:
            data_path = os.path.join(dir, 'tmpvocab.model')
            with open(data_path, 'w') as f:
                if isinstance(tokens, Counter):
                    for token, count in tokens.items():
                        for _ in range(count):
                            f.write(token + '\n')
                else:
                    for element in tokens:
                        if isinstance(element, str):
                            f.write(element + '\n')
                        else:
                            f.write(' '.join(element))
                            f.write('\n')
            return self.create_vocabulary_from_file(data_path)   

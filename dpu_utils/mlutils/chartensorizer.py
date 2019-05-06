import numpy as np

from typing import Optional

__all__ = ['CharTensorizer']


class CharTensorizer:
    """Tensorize strings into characters"""

    def __init__(self, max_num_chars: Optional[int], lower_case_all: bool, include_space: bool):
        self.__max_num_chars = max_num_chars
        self.__lower_case_all = lower_case_all

        self.__ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
        if not self.__lower_case_all:
            self.__ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + self.__ALPHABET
        if include_space:
            self.__ALPHABET += ' '

        self.__ALPHABET_DICT = {char: idx + 2 for (idx, char) in enumerate(self.__ALPHABET)}  # "0" is PAD, "1" is UNK
        self.__ALPHABET_DICT['PAD'] = 0
        self.__ALPHABET_DICT['UNK'] = 1

    @property
    def max_char_length(self) -> Optional[int]:
        return self.__max_num_chars

    def num_chars_in_vocabulary(self)-> int:
        return len(self.__ALPHABET_DICT)

    def __get_char_idx(self, character: str) -> int:
        idx = self.__ALPHABET_DICT.get(character)
        if idx is not None:
            return idx
        return self.__ALPHABET_DICT['UNK']

    def get_word_from_ids(self, ids: np.ndarray)-> str:
        return ''.join(self.__ALPHABET[i] if i!=1 else '<UNK>' for i in ids)

    def tensorize_str(self, input: str) -> np.ndarray:
        if self.__lower_case_all:
            input = input.lower()

        def char_iterator():
            for i, c in enumerate(input):
                if self.__max_num_chars is not None and i >= self.__max_num_chars:
                    break
                yield self.__get_char_idx(c)
            if self.__max_num_chars is not None and len(input) < self.__max_num_chars:
                pad_id = self.__get_char_idx('PAD')
                yield from (pad_id for _ in range(self.__max_num_chars - len(input)))
        return np.fromiter(char_iterator(), dtype=np.uint8)

import logging
import os
from typing import List, Optional
import sentencepiece as spm

logger = logging.getLogger(__name__)

SPIECE_UNDERLINE = u'▁'

class BpeVocabulary:

    """
    A vocabulary that maps strings to unique ids (and back), and a tokenizer based on
    Byte-Pair encoding.

    To create a vocabulary use `BpeVocabulary.create_vocabulary()` and pass
    a path to plain text file containing input data. The control flow symbols have to be introduced
    manually, during preprocessing step.

    Vocabulary object usage: Assuming an initialized vocabulary `v`:

       * To get the id of an element use `v.get_id_or_unk("element")`.
       * To get the ids of a sequence, use `v.get_id_or_unk_multiple(..)`.
       * To get the size of the vocabulary use `len(v)`
    """

    def __init__(self, max_size: int, vocab_file: str=None,
                 bos_token: str="<s>", eos_token: str="</s>", unk_token: str="<unk>", pad_token: str="<pad>", 
                 user_defined_symbols: List[str]=["<DEDENT>", "<INDENT>", "<NUM_LIT>", "<STR_LIT>"],
                 control_symbols: List[str]=["<endofline>", "<endoftext>"]) -> None:

        self.max_size=max_size
        self.bos_token=bos_token
        self.eos_token=eos_token
        self.unk_token=unk_token
        self.pad_token=pad_token
        self.vocab_file = vocab_file
        self.user_defined_symbols=",".join(user_defined_symbols)
        self.control_symbols=",".join(control_symbols)

        self.__sp_model = spm.SentencePieceProcessor()
        if vocab_file is not None:
            self.__sp_model.Load(vocab_file)

    def pad_token(self):
        """ Get padding token (string)"""
        if self.pad_token is None:
            logger.error("Using pad_token, but it is not set yet.")
        return self.pad_token

    def unk_token(self):
        """ Get unknown token (string) """
        if self.unk_token is None:
            logger.error("Using unk_token, but it is not set yet.")
        return self.unk_token

    def vocab_size(self):
        return len(self.__sp_model)

    def tokenize(self, text: str) -> List[str]:
        """ Tokenize a string. """
        pieces = self.__sp_model.EncodeAsPieces(text)

        new_pieces = []
        for piece in pieces:
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


    def get_id_or_unk(self, token: str) -> int:
        return self._convert_token_to_id(token)

    def get_id_or_unk_multiple(self, tokens: List[str], pad_to_size: Optional[int] = None,
                               padding_element: int = 0) -> List[int]:
        if pad_to_size is not None:
            tokens = tokens[:pad_to_size]

        ids = [self.get_id_or_unk(t) for t in tokens]

        if pad_to_size is not None and len(ids) != pad_to_size:
            ids += [padding_element] * (pad_to_size - len(ids))

        return ids

    def get_name_for_id(self, token_id: int) -> str:
        return self._convert_id_to_token(token_id)

    def _convert_token_to_id(self, token: str) -> int:
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.__sp_model.PieceToId(token)

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        token = self.__sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = ''.join(tokens).replace(SPIECE_UNDERLINE, ' ').strip()
        return out_string

    def create_vocabulary(self, sp_text_file: str, num_threads: int=40, max_sentence_length: int=16384, character_coverage: float=0.99995) -> None:
        """
        Train sentencepiece tokenizer using BPE model and build a vocabulary.

        sp_text_file: path to a plain text file containing the training dataset 
        """

        command = [
                    f"--input={sp_text_file}",
                    f"--num_threads={num_threads}",
                    f"--model_prefix=bpe_{self.max_size}",
                    f"--vocab_size={self.max_size}",
                    f"--model_type=bpe",
                    f"--max_sentence_length={max_sentence_length}",
                    f"--bos_id={self.bos_token}",
                    f"--eos_id={self.eos_token}",
                    f"--pad_id={self.eos_token}",                    
                    f"--unk_piece={self.unk_token}",
                    f"--user_defined_symbols={self.user_defined_symbols}",
                    f"--control_symbols={self.control_symbols}",
                    f"--character_coverage={character_coverage}",
                ]

        spm.SentencePieceTrainer.train(
            " ".join(command)
        )

        assert self.__sp_model.Load(f"bpe_{self.max_size}.model")

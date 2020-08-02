from typing import List, Optional, Dict, Set
from abc import ABC, abstractmethod


class OmniTokenizer(ABC):
    def __init__(self, bos_token, bos_idx, eos_token, eos_idx, pad_token, pad_idx, unk_token, unk_idx, mask_token,
                 mask_idx):

        self.bos_token = bos_token
        self.bos_idx = bos_idx

        self.eos_token = eos_token
        self.eos_idx = eos_idx

        self.pad_token = pad_token
        self.pad_idx = pad_idx

        self.unk_token = unk_token
        self.unk_idx = unk_idx

        self.mask_token = mask_token
        self.mask_idx = mask_idx

    @property
    def dummy_input(self):
        """
        input to the model just to try
        """
        pass

    @abstractmethod
    def vocab_size(self):
        """
        get the vocabulary size of the tokenizer
        """
        ...

    @abstractmethod
    def get_vocab(self) -> Set:
        """
        get the vocabulary
        :return:
        """

        ...

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        tokenizer the input string according to the tokenizer strategy
        :param text: text to tokenizer
        :return:
        """

        ...

    @abstractmethod
    def _convert_id_to_token(self, index: int) -> str:
        ...

    @abstractmethod
    def _convert_token_to_id(self, token: str) -> int:
        ...

    def convert_ids_to_tokens(self, ids: List[int], skip_special_tokens=False) -> List[str]:
        """
        convert a single sequence of indices in a list of tokens
        :param ids: ids to convert
        :param skip_special_tokens: if yes, remove special tokens
        :return:
        """
        if skip_special_tokens:
            ids = filter(lambda x: x not in self.special_ids, ids)
        return [self._convert_id_to_token(id) for id in ids]

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self._convert_token_to_id(token) for token in tokens]

    @abstractmethod
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        ...

    @abstractmethod
    def encode(self,
               text: str,
               text_pair: Optional[str] = None,
               add_special_tokens: bool = True,
               return_position_ids: bool = False,
               max_length: Optional[int] = 512
               ) -> Dict:
        """
        encode the given text or text pair
        :param text: first utterance
        :param text_pair: second utterance
        :param add_special_tokens: if yes, special tokens are added
        :param return_position_ids: return position ids for positional encoding
        :param max_length: max sequence length
        :return:
        """

        ...

    def decode(self, tokens_ids: List[int], skip_special_tokens=False) -> str:
        """
        decode input tokens
        :param tokens_ids: input tokens
        :param skip_special_tokens: if yes, special tokens are removed
        :return:
        """
        filtered_tokens = self.convert_ids_to_tokens(tokens_ids,
                                                     skip_special_tokens=skip_special_tokens)
        text = self.convert_tokens_to_string(filtered_tokens)
        return text

    @classmethod
    def load(cls, path_: str):
        raise NotImplementedError
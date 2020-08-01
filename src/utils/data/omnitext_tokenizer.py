from typing import List, Optional, Dict
from abc import ABC, abstractmethod


BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
MASK_TOKEN = "<msk>"

BOS_IDX = 0
EOS_IDX = 2
PAD_IDX = 1
UNK_IDX = 3
MASK_IDX = 250001

DUMMY_INPUT = [[0, 25378, 8999, 38, 2]]


class OmniTokenizer(ABC):
    def __init__(self, bos_token=BOS_TOKEN, bos_idx=BOS_IDX, eos_token=EOS_TOKEN, eos_idx=EOS_IDX,
                 pad_token=PAD_TOKEN, pad_idx=PAD_IDX, unk_token=UNK_TOKEN, unk_idx=UNK_IDX,
                 mask_token=MASK_TOKEN, mask_idx=MASK_IDX):

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

    @abstractmethod
    @property
    def vocab_size(self):
        """
        get the vocabulary size of the tokenizer
        :return:
        """

        ...

    @abstractmethod
    def get_vocab(self) -> Dict:
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
    def convert_ids_to_tokens(self, ids: List[int], skip_special_tokens=False) -> List[str]:
        """
        convert a single sequence of indices in a list of tokens
        :param ids: ids to convert
        :param skip_special_tokens: if yes, remove special tokens
        :return:
        """

        ...

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
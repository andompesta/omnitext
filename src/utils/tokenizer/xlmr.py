from __future__ import absolute_import

from src.utils.tokenizer import OmniTokenizer
from sentencepiece import SentencePieceProcessor
from typing import Dict, List, Set, Optional
from itertools import chain

SPIECE_UNDERLINE = "▁"
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


class Tokenizer(OmniTokenizer):
    def __init__(self, path_: str):
        super(Tokenizer, self).__init__(
            bos_token=BOS_TOKEN,
            bos_idx=BOS_IDX,
            eos_token=EOS_TOKEN,
            eos_idx=EOS_IDX,
            pad_token=PAD_TOKEN,
            pad_idx=PAD_IDX,
            unk_token=UNK_TOKEN,
            unk_idx=UNK_IDX,
            mask_token=MASK_TOKEN,
            mask_idx=MASK_IDX
        )
        self.spm = SentencePieceProcessor()
        self.spm.Load(path_)
        self.vocab_path = path_

        self.special_tokens_to_ids = {
            self.bos_token: self.bos_idx,
            self.eos_token: self.eos_idx,
            self.pad_token: self.pad_idx,
            self.unk_token: self.unk_idx,
            self.mask_token: self.mask_idx
        }

        self.special_ids_to_tokens = dict([(k, v) for (v, k) in self.special_tokens_to_ids.items()])
        self.special_tokens = set(self.special_tokens_to_ids.keys())
        self.special_ids = set(self.special_ids_to_tokens.keys())

        self.special_offset = 1
        self.position_offset = 2


    def __getstate__(self):
        state = self.__dict__.copy()
        state["spm"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.spm = SentencePieceProcessor()
        self.spm.Load(self.vocab_path)


    @property
    def dummy_input(self):
        return DUMMY_INPUT

    @classmethod
    def load(cls, path_: str):
        raise cls(path_)

    def _convert_id_to_token(self, index: int) -> str:
        if index in self.special_ids:
            return self.special_ids_to_tokens[index]
        return self.spm.IdToPiece(index - self.special_offset)

    def _convert_token_to_id(self, token: str) -> int:
        if token in self.special_tokens:
            return self.special_tokens_to_ids[token]

        spm_id = self.spm.PieceToId(token)
        return spm_id + self.special_offset if spm_id else self.unk_idx

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()

    def tokenize(self, text: str) -> List[str]:
        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                sub_text = sub_text.rstrip()
                if i == 0 and not sub_text:
                    result += [tok]
                elif i == len(split_text) - 1:
                    if sub_text:
                        result += [sub_text]
                    else:
                        pass
                else:
                    if sub_text:
                        result += [sub_text]
                    result += [tok]
            return result

        def split_on_tokens(tok_list: Set[str], text: str) -> List[str]:
            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.special_tokens:
                        tokenized_text += split_on_token(tok, sub_text)
                    else:
                        tokenized_text += [sub_text]
                text_list = tokenized_text

            return list(chain.from_iterable((self.spm.EncodeAsPieces(token) if token not in self.special_tokens
                                             else [token] for token in tokenized_text)))

        tokenized_text = split_on_tokens(self.special_tokens, text)
        return tokenized_text

    def build_inputs_with_special_tokens(self,
                                         ids: List[int],
                                         pair_ids: Optional[List[int]] = None) -> List[int]:
        ret = [self.bos_idx] + ids + [self.eos_idx]
        if pair_ids is not None:
            ret += [self.eos_idx] + pair_ids + [self.eos_idx]
        return ret

    def vocab_size(self):
        return len(self.spm) + self.special_offset + 1 # add mask token

    def get_vocab(self) -> Dict:
        vocab = dict([(self._convert_id_to_token(id), id) for id in range(self.vocab_size())])
        return vocab

    def truncate_sequences(self, ids: List[int], pari_ids: Optional[List[int]], num_token_to_remove: int = 0):
        if num_token_to_remove <= 0:
            return ids, pari_ids

        if pari_ids is None or len(ids) > len(pari_ids):
            ids = ids[:-num_token_to_remove]
        else:
            pari_ids = pari_ids[:-num_token_to_remove]

        return ids, pari_ids

    def encode(self,
               text: str,
               text_pair: Optional[str] = None,
               add_special_tokens: bool = True,
               return_position_ids: bool = False,
               max_length: Optional[int] = 512
               ) -> Dict:
        def get_input_ids(text: str):
            if isinstance(text, str):
                tokens = self.tokenize(text)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                return self.convert_tokens_to_ids(text)

        ids = get_input_ids(text)
        pair_ids = get_input_ids(text_pair) if text_pair is not None else None

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        # handle max length
        total_len = len_ids + len_pair_ids
        num_special_tokens = 4 if pair_ids else 2
        if max_length and total_len > (max_length - num_special_tokens):
            ids, pair_ids = self.truncate_sequences(ids, pair_ids, total_len - (max_length - num_special_tokens))

        # handle special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids

        if return_position_ids:
            position_ids = list(range(self.position_offset, len(sequence) + self.position_offset))
            encode_input = dict(input_ids=sequence, position_ids=position_ids)
        else:
            encode_input = dict(input_ids=sequence)

        return encode_input


if __name__ == '__main__':
    import torch
    import os
    from dynaconf import settings

    xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large')
    tok = Tokenizer(os.path.join(settings.get("ckp_dir"), "import", "sentencepiece.bpe.model"))

    en_tokens = xlmr.encode('Hello world!')
    assert en_tokens.tolist() == [0, 35378, 8999, 38, 2]

    input_ids = tok.encode('Hello world!')["input_ids"]
    assert en_tokens.tolist() == input_ids
    print(input_ids)

    input_ids = tok.encode('<msk>')["input_ids"]
    print(input_ids)

    input_ids = tok.encode('ciao <msk> mondo')["input_ids"]
    print(input_ids)
    print(tok.decode(input_ids, skip_special_tokens=False))
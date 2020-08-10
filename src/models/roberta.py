import torch
from torch import nn, Tensor
from typing import Optional, Tuple

from src.config import XLMRobertaConfig
from src.modules import (
    Encoder,
    BaseModel
)
from src.modules.utils import ACT_NAME_TO_FN


class RobertaEncoder(Encoder):
    def __init__(self, conf: XLMRobertaConfig):
        super(RobertaEncoder, self).__init__(conf)

    def forward(
            self,
            input_ids: Tensor,
            position_ids: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
    ):
        if attention_mask is None:
            # do not attend of pad_idx
            attention_mask = input_ids.eq(self.pad_idx)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)

        outputs = super(RobertaEncoder, self).forward(
            input_ids=input_ids,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            position_ids=position_ids
        )
        return outputs

    def get_extended_attention_mask(
            self,
            attention_mask: Tensor
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for attention_mask (shape {})".format(attention_mask.shape)
            )

        # Since attention_mask is 0.0 for positions we want to attend and 1.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = extended_attention_mask * -10000.0
        return extended_attention_mask

class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, conf: XLMRobertaConfig):
        super().__init__()
        self.hidden_size = conf.hidden_size
        self.layer_norm_eps = conf.layer_norm_eps
        self.output_dim = conf.vocab_size

        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation_fn = ACT_NAME_TO_FN[conf.hidden_act]
        self.layer_norm = nn.LayerNorm(self.hidden_size,
                                       eps=self.layer_norm_eps, elementwise_affine=True)

        self.decoder = nn.Linear(self.hidden_size, self.output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        self.decoder.bias = self.bias

    def forward(
            self,
            features: Tensor,
    ) -> Tensor:
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = self.decoder(x)
        return x


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, conf: XLMRobertaConfig):
        super().__init__()
        self.hidden_size = conf.hidden_size
        self.num_labels = conf.num_labels

        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(p=conf.hidden_dropout_prob)
        self.out_proj = nn.Linear(self.hidden_size, self.num_labels)

    def forward(
            self,
            features: Tensor
    ) -> Tensor:
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class XlmMaskedLanguageModel(BaseModel):
    def __init__(self, conf: XLMRobertaConfig):
        conf.name += "-mlm"
        super(XlmMaskedLanguageModel, self).__init__(conf)

        self.sentence_encoder = RobertaEncoder(conf)
        self.lm_head = RobertaLMHead(conf)

        self.tie_weights()
        self.init_weights()

    def forward(
            self,
            input_ids: Tensor,
            position_ids: Optional[Tensor],
            attention_mask: Optional[Tensor] = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tuple[Tensor]], Optional[Tuple[Tensor]]]:

        enc_outputs = self.sentence_encoder(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        hidden_state = enc_outputs[0]
        logits = self.lm_head(hidden_state)

        outputs = (logits, hidden_state, enc_outputs[1], enc_outputs[2])
        return outputs

    def _init_weights(self, module: torch.nn.Module):
        if hasattr(module, "_init_weights"):
            module._init_weights()
            print("_init_weights for {}".format(module))

        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
            print("linear init for {}".format(module))

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

            print("embedding init for {}".format(module))

        else:
            print("----> WARNING: module {} not initialized".format(module))

    def get_input_embeddings(self):
        return self.sentence_encoder.embed_tokens

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.lm_head.decoder


class XlmClassificationModel(BaseModel):
    def __init__(self, conf: XLMRobertaConfig):
        conf.name += "-classify"
        super(XlmClassificationModel, self).__init__(conf)
        self.sentence_encoder = RobertaEncoder(conf)
        self.classification_head = RobertaClassificationHead(conf)

        self.init_weights()

    def forward(
            self,
            input_ids: Tensor,
            position_ids: Optional[Tensor],
            attention_mask: Optional[Tensor] = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tuple[Tensor]], Optional[Tuple[Tensor]]]:

        enc_outputs = self.sentence_encoder(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        hidden_state = enc_outputs[0]
        logits = self.classification_head(hidden_state)

        outputs = (logits, hidden_state, enc_outputs[1], enc_outputs[2])
        return outputs

    def _init_weights(self, module: torch.nn.Module):
        if hasattr(module, "_init_weights"):
            module._init_weights()
            print("_init_weights for {}".format(module))

        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
            print("linear init for {}".format(module))

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

            print("embedding init for {}".format(module))

        else:
            print("----> WARNING: module {} not initialized".format(module))

    def get_input_embeddings(self):
        return self.sentence_encoder.embed_tokens

    def get_output_embeddings(self) -> torch.nn.Module:
        raise ValueError("classification model has no output embedding")





import torch
from torch import nn, Tensor
from typing import Optional, Tuple
from collections import OrderedDict

from src.config import RobertaConfig

from src.modules import (
    Encoder,
    BaseModel,
    ClassificationHead,
    LMHead
)


class RobertaEncoder(Encoder):
    def __init__(self, conf: RobertaConfig):
        super(RobertaEncoder, self).__init__(conf)

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            **kwargs
    ):
        if attention_mask is None:
            if conf.attention_type == "full":
                attention_mask = input_ids.eq(self.pad_idx)
                # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
                # ourselves in which case we just need to make it broadcastable to all heads.
                attention_mask = self.get_extended_attention_mask(attention_mask)
            elif conf.attention_type == "linear":
                attention_mask = input_ids.ne(self.pad_idx)
                attention_mask = self.get_linear_attention_mask(attention_mask)
            else:
                raise NotImplementedError('Not implemented yet. Support only "full" and "linear".')

        outputs = super(RobertaEncoder, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs
        )
        return outputs

    def get_linear_attention_mask(
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

        # linear attention does not support causal  computation this only a 2 dimensional matrix is accepted
        assert attention_mask.dim() == 2

        attention_mask = attention_mask[:, None, :, None].to(dtype=self.dtype)
        return attention_mask


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


class RobertaMaskedLanguageModel(RobertaEncoder, BaseModel):
    def __init__(self, conf: RobertaConfig):
        conf.name += "-mlm"
        super(RobertaMaskedLanguageModel, self).__init__(conf)
        self.lm_head = LMHead(conf)
        self.tie_weights()
        self.init_weights()

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            **kwargs
    ) -> Tuple[Tensor, Tensor]:

        enc_outputs = self.sentence_encoder(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        logits = self.lm_head(enc_outputs)

        outputs = (logits, enc_outputs)
        return outputs

    def _init_weights(self, module: torch.nn.Module):
        if hasattr(module, "_init_weights") and not isinstance(module, RobertaMaskedLanguageModel):
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

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant(module.bias, 0.)
            nn.init.constant(module.weight, 1.)
            print("layer norm init for {}".format(module))

        else:
            print("----> WARNING: module {} not initialized".format(module))


    def get_input_embeddings(self):
        return self.sentence_encoder.embed_tokens

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.lm_head.decoder


class RobertaClassificationModel(RobertaEncoder, BaseModel):
    def __init__(self, conf: RobertaConfig):
        conf.name += "-classify"
        super(RobertaClassificationModel, self).__init__(conf)
        self.classification_head = ClassificationHead(conf)

        self.init_weights()

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            **kwargs
    ) -> Tuple[Tensor, Tensor]:

        enc_outputs = self.sentence_encoder(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        logits = self.classification_head(enc_outputs)

        outputs = (logits, enc_outputs)
        return outputs

    def _init_weights(self, module: torch.nn.Module):
        if hasattr(module, "_init_weights") and not isinstance(module, RobertaClassificationModel):
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

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.)
            nn.init.constant_(module.weight, 1.)
            print("layer norm init for {}".format(module))

        else:
            print("----> WARNING: module {} not initialized".format(module))

    def get_input_embeddings(self):
        return self.sentence_encoder.embed_tokens

    def get_output_embeddings(self) -> torch.nn.Module:
        raise ValueError("classification model has no output embedding")

# class XLMClassificationModel(BaseModel):
#     def __init__(self, conf: XLMRobertaConfig):
#         conf.name += "-classify"
#         super(XLMClassificationModel, self).__init__(conf)
#         self.sentence_encoder = RobertaEncoder(conf)
#         self.classification_head = RobertaClassificationHead(conf)
#
#         self.init_weights()
#
#     def forward(
#             self,
#             input_ids: Tensor,
#             position_ids: Optional[Tensor],
#             attention_mask: Optional[Tensor] = None,
#             output_attentions: bool = False,
#             output_hidden_states: bool = False,
#     ) -> Tuple[Tensor, Tensor, Optional[Tuple[Tensor]], Optional[Tuple[Tensor]]]:
#
#         enc_outputs = self.sentence_encoder(
#             input_ids,
#             position_ids=position_ids,
#             attention_mask=attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states
#         )
#         hidden_state = enc_outputs[0]
#         logits = self.classification_head(hidden_state)
#
#         outputs = (logits, hidden_state, enc_outputs[1], enc_outputs[2])
#         return outputs
#
#     def _init_weights(self, module: torch.nn.Module):
#         if hasattr(module, "_init_weights") and not isinstance(module, XLMClassificationModel):
#             module._init_weights()
#             print("_init_weights for {}".format(module))
#
#         elif isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#             print("linear init for {}".format(module))
#
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#
#             print("embedding init for {}".format(module))
#
#         elif isinstance(module, nn.LayerNorm):
#             nn.init.constant_(module.bias, 0.)
#             nn.init.constant_(module.weight, 1.)
#             print("layer norm init for {}".format(module))
#
#         else:
#             print("----> WARNING: module {} not initialized".format(module))
#
#     def get_input_embeddings(self):
#         return self.sentence_encoder.embed_tokens
#
#     def get_output_embeddings(self) -> torch.nn.Module:
#         raise ValueError("classification model has no output embedding")



if __name__ == '__main__':
    import os
    from dynaconf import settings
    from transformers import RobertaModel
    import re

    conf = RobertaConfig(num_labels=2)
    roberta = RobertaClassificationModel(conf)

    hugging_face = RobertaModel.from_pretrained('roberta-import', return_dict=True)


    state_dict = hugging_face.state_dict()

    # fix pre-trained names
    state_dict.pop('embeddings.token_type_embeddings.weight')
    state_dict.pop("embeddings.position_ids")

    # rename dict
    renamed_dict = OrderedDict()
    for name, value in state_dict.items():
        if name.startswith("encoder."):
            name = name.replace("encoder.", "")
        elif name.startswith("pooler."):
            name = name.replace("pooler.", "classification_head.")
        elif name.startswith("embeddings."):
            name = name.replace("embeddings.", "", 1)
            if name.startswith("word_embeddings"):
                name = name.replace("word_embeddings", "embed_tokens")

            if name.startswith("position_embeddings"):
                name = name.replace("position_embeddings", "embed_positions")

            if name.startswith("LayerNorm"):
                name = name.replace("LayerNorm", "embed_layer_norm")

        if ".LayerNorm." in name:
            name = name.replace(".LayerNorm.", ".layer_norm.")

        if "attention.self" in name:
            name = name.replace("attention.self", "attn")

        if "attention.output" in name:
            name = name.replace("attention.output", "self_output")

        renamed_dict[name] = value

    roberta.load_state_dict(renamed_dict, strict=False)
    roberta.save(os.path.join(settings.get("exp_dir"), "pre-trained"), file_name="roberta-import.pth.tar", is_best=False)
    roberta.load_state_dict(renamed_dict, strict=True)

import torch
from torch import nn, Tensor
from typing import Optional, Tuple
from collections import OrderedDict

from src.config import RobertaConfig

from src.modules import (
    BaseModel,
    TokenEmbedding,
    PositionalEmbedding,
    Encoder,
    ClassificationHead,
    LMHead,
    Pooler
)


class RobertaModel(BaseModel):
    def __init__(self, conf: RobertaConfig):
        super(RobertaModel, self).__init__(conf)
        self.embedding_scale = conf.embedding_scale

        self.embed_tokens = TokenEmbedding(
            num_embeddings=conf.vocab_size,
            embedding_dim=conf.hidden_size,
            padding_idx=conf.pad_token_id,
            initializer_range=conf.initializer_range,
            scale=conf.embedding_scale
        )

        self.embed_positions = PositionalEmbedding(
            num_embeddings=conf.max_position_embeddings,
            embedding_dim=conf.hidden_size,
            padding_idx=conf.pad_token_id,
            initializer_range=conf.initializer_range,
            type=conf.pos_embeddings_type,
            trainable_num_embeddings=conf.trainable_num_embeddings,
            fixed_num_embeddings=conf.fixed_num_embeddings
        )

        self.embed_layer_norm = nn.LayerNorm(
            conf.hidden_size,
            eps=conf.layer_norm_eps,
            elementwise_affine=True
        )

        self.embed_dropout = nn.Dropout(conf.hidden_dropout_prob)

        self.encoder = Encoder(conf)
        self.pooler = Pooler(conf)
        self.dtype = next(self.parameters()).dtype

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            **kwargs
    ) -> Tuple[Tensor, Tensor]:
        if attention_mask is None:
            if self.conf.attention_type == "full":
                attention_mask = input_ids.eq(self.conf.pad_token_id)
                # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
                # ourselves in which case we just need to make it broadcastable to all heads.
                attention_mask = self.get_extended_attention_mask(attention_mask)
            elif conf.attention_type == "linear":
                attention_mask = input_ids.ne(self.conf.pad_token_id)
                attention_mask = self.get_linear_attention_mask(attention_mask)
            else:
                raise NotImplementedError('Not implemented yet. Support only "full" and "linear".')

        hidden_states = self.embeddings(
            input_ids,
            position_ids
        )

        hidden_states = self.encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs
        )

        pooled_state = self.pooler(hidden_states)

        return pooled_state, hidden_states

    def embeddings(
            self,
            input_ids: Tensor,
            position_ids: Optional[Tensor] = None
    ) -> Tensor:
        hidden_states = self.embed_tokens(input_ids)
        if self.embedding_scale is not None:
            hidden_states *= self.embedding_scale

        hidden_states += self.embed_positions(
            input_ids,
            position_ids
        )

        hidden_states = self.embed_layer_norm(hidden_states)
        hidden_states = self.embed_dropout(hidden_states)

        return hidden_states

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
            print("linear init for {}".format(module))

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.)
            nn.init.constant_(module.weight, 1.)
            print("layer norm init for {}".format(module))

        elif hasattr(module, "_init_weights") and not isinstance(module, BaseModel):
            module._init_weights()

        else:
            print("----> WARNING: module {} not initialized".format(module))

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

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self) -> torch.nn.Module:
        raise ValueError("classification model has no output embedding")

class RobertaMaskedLanguageModel(RobertaModel):
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

        pooled_state, hidden_states = super(RobertaMaskedLanguageModel, self).forward(
            input_ids,
            attention_mask,
            position_ids,
            **kwargs
        )

        logits = self.lm_head(hidden_states)

        return logits

    def get_input_embeddings(self):
        return self.sentence_encoder.embed_tokens

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.lm_head.decoder


class RobertaClassificationModel(RobertaModel):
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

        pooled_state, hidden_states = super(RobertaClassificationModel, self).forward(
            input_ids,
            attention_mask,
            position_ids,
            **kwargs
        )

        logits = self.classification_head(pooled_state)

        return logits


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

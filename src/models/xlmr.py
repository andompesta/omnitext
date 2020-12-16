from torch import nn, Tensor
from typing import Optional, Tuple
from src.modules import (Pooler, ClassificationHead, Adapter, BaseModel)
from src.models.roberta import RobertaModel
from src.config import XLMRobertaConfig



class XLMRobertaClassificationModel(RobertaModel):
    def __init__(self, conf: XLMRobertaConfig):
        conf.name += "-classify"
        super(XLMRobertaClassificationModel, self).__init__(conf)

        if hasattr(conf, "adapter_size"):
            self.adapter = Adapter(conf)
        self.classification_head = ClassificationHead(conf)

        self.init_weights()

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            **kwargs
    ) -> Tuple[Tensor, Tensor]:

        pooled_state, hidden_states = super(XLMRobertaClassificationModel, self).forward(
            input_ids,
            attention_mask,
            position_ids,
            **kwargs
        )

        if hasattr(self, "adapter"):
            pooled_state = self.adapter(pooled_state)

        logits = self.classification_head(pooled_state)

        return logits

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

    def get_input_embeddings(self):
        return self.sentence_encoder.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        raise ValueError("classification model has no output embedding")

if __name__ == '__main__':
    import os
    from dynaconf import settings
    from transformers import XLMRobertaModel
    from collections import OrderedDict
    import re

    conf = XLMRobertaConfig(num_labels=2)
    xlmr = XLMRobertaClassificationModel(conf)

    hugging_face = XLMRobertaModel.from_pretrained("xlm-roberta-base")


    state_dict = hugging_face.state_dict()

    # fix pre-trained names
    state_dict.pop('embeddings.token_type_embeddings.weight')
    state_dict.pop("embeddings.position_ids")

    # rename dict
    renamed_dict = OrderedDict()
    for name, value in state_dict.items():

        if name == "embeddings.word_embeddings.weight":
            name = "embed_tokens.weight"

        elif name == 'embeddings.position_embeddings.weight':
            name = 'embed_positions.weight'

        elif name == 'embeddings.LayerNorm.weight':
            name = 'embed_layer_norm.weight'

        elif name == 'embeddings.LayerNorm.bias':
            name = 'embed_layer_norm.bias'

        elif ".attention.self." in name:
            name = name.replace(".attention.self.", ".attn.")
        elif ".attention.output." in name:
            name = name.replace(".attention.output.", ".self_output.")

        if ".LayerNorm." in name:
            name = name.replace(".LayerNorm.", ".layer_norm.")

        renamed_dict[name] = value

    print(xlmr.load_state_dict(renamed_dict, strict=False))
    xlmr.save(os.path.join(settings.get("ckp_dir"), "import"), file_name="xlm-roberta-pre-trained.pth.tar", is_best=False)

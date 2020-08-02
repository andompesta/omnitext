""" XLM-RoBERTa configuration """

from .base import BaseConf

class XLMRobertaConfig(BaseConf):
    """
    This class overrides :class:`~transformers.RobertaConfig`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    @property
    def name(self) -> str:
        return "xlm-roberta"

    def __init__(
            self,
            vocab_size=250002,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=.1,
            attention_probs_dropout_prob=.1,
            max_position_embeddings=514,
            initializer_range=.02,
            layer_norm_eps=.00001,
            output_attentions=False,
            output_hidden_states=False,
            **kwargs
    ):
        super(XLMRobertaConfig, self).__init__(d_model=hidden_size,
                                               n_head=num_attention_heads,
                                               n_layer=num_hidden_layers)
        self.vocab_size = vocab_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

        if "num_labels" in kwargs:
            self.num_labels = kwargs.get("num_labels")

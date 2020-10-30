""" XLM-RoBERTa configuration """

from src.config.base import BaseConf

class RobertaConfig(BaseConf):
    """
    This class overrides :class:`~transformers.RobertaConfig`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    def __init__(
            self,
            name="roberta-pre-trained",
            vocab_size=50265,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=.1,
            attention_probs_dropout_prob=.1,
            intermediate_dropout_prob=0.,
            max_position_embeddings=512,
            initializer_range=.02,
            layer_norm_eps=0.00001,
            url="https://dl.fbaipublicfiles.com/fairseq/models/xlmr.base.tar.gz",
            pad_idx=1,
            bos_idx=0,
            eos_idx=2,
            embedding_scale=None,
            learned_pos_embeddings=True,
            attention_type="full",
            kernel_type="favor",
            kernel_size=256,
            **kwargs
    ):
        super(RobertaConfig, self).__init__(name=name)
        self.vocab_size = vocab_size
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.intermediate_dropout_prob = intermediate_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.url = url
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.embedding_scale = embedding_scale
        self.learned_pos_embeddings = learned_pos_embeddings
        self.attention_type = attention_type
        self.kernel_type = kernel_type
        self.kernel_size = kernel_size

        if "num_labels" in kwargs:
            self.num_labels = kwargs.get("num_labels")


if __name__ == '__main__':
    from transformers import RobertaTokenizerFast
    conf = RobertaConfig()

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    assert conf.bos_idx == tokenizer.bos_token_id
    assert conf.pad_idx == tokenizer.pad_token_id
    assert conf.eos_idx == tokenizer.eos_token_id

    print(tokenizer(["hello word", "how are you"], return_attention_mask=False))
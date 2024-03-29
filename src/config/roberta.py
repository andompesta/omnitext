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
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            embedding_scale=None,
            pos_embeddings_type="learned",
            trainable_num_embeddings=2488,
            fixed_num_embeddings=514,
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
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.embedding_scale = embedding_scale
        self.pos_embeddings_type = pos_embeddings_type
        self.trainable_num_embeddings = trainable_num_embeddings
        self.fixed_num_embeddings = fixed_num_embeddings
        self.attention_type = attention_type
        self.kernel_type = kernel_type
        self.kernel_size = kernel_size


        for n, v in kwargs.items():
            setattr(self, n, v)


if __name__ == '__main__':
    from transformers import RobertaTokenizerFast
    conf = RobertaConfig()

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-import")

    assert conf.bos_token_id == tokenizer.bos_token_id
    assert conf.pad_token_id == tokenizer.pad_token_id
    assert conf.eos_token_id == tokenizer.eos_token_id

    print(tokenizer(["hello word", "how are you"], return_attention_mask=False))
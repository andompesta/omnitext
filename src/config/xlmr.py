""" XLM-RoBERTa configuration """

from src.config.roberta import RobertaConfig

class XLMRobertaConfig(RobertaConfig):
    """
    This class overrides :class:`~transformers.RobertaConfig`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    def __init__(
            self,
            name="xlm-roberta-pre-trained",
            vocab_size=250002,
            max_position_embeddings=514,
            **kwargs
    ):
        super(XLMRobertaConfig, self).__init__(
            name=name,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            url="http://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz",
            **kwargs
        )


if __name__ == '__main__':
    from transformers import XLMRobertaTokenizer
    conf = XLMRobertaConfig()

    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-import")

    assert conf.bos_token_id == tokenizer.bos_token_id
    assert conf.pad_token_id == tokenizer.pad_token_id
    assert conf.eos_token_id == tokenizer.eos_token_id

    print(tokenizer(["hello word", "how are you"], return_attention_mask=False))

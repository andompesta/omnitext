""" XLM-RoBERTa configuration """

from .roberta import RobertaConfig

class XLMRobertaConfig(RobertaConfig):
    """
    This class overrides :class:`~transformers.RobertaConfig`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    def __init__(
            self,
            name="xlm-roberta-pre-trained",
            vocab_size=50265,
            **kwargs
    ):
        super(XLMRobertaConfig, self).__init__(
            name=name,
            vocab_size=vocab_size,
            url="http://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz",
            **kwargs
        )
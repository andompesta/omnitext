import re
from sacremoses import MosesPunctNormalizer, MosesTokenizer, MosesDetokenizer
from typing import List, Any, Union, Tuple

PROTECTED_PATTENRS = [
    r"([a-z0-0_.+-]+@[a-z0-9-]+\.[a-z0-9-.]+)",
    r"(?:(?:https?|ftp|file):(//|t\.)|www\.|ftp\.)(?:([-A-Z0-9+&@#/%=~_|$?!:,.]*)|[-A-Z0-9+&@#/%=~_|$?!:,.])*(?:([-A-Z0-9+&@#/%=~_|$?!:,.]*)|[A-Z0-9+&@#/%=~_|$])"
]


def remove_non_utf8(text: str) -> str:
    return bytes(text, "utf8").decode("utf-8", "ignore")

def remove_double_spaces(text: str) -> str:
    return re.sub(r"\s\s+", " ", text).strip()

def moses_clean_sentences(
        text: str,
        lang: str = "en",
        lowercase: bool = True,
        protected_patterns: Union[List[str], Tuple[str]] = tuple(PROTECTED_PATTENRS)
) -> str:
    """
    perform moses preprocessing
    """
    normalizer = MosesPunctNormalizer(
        lang=lang,
        pre_replace_unicode_punct=True,
        post_remove_control_chars=True
    )
    tokenizer = MosesTokenizer(lang=lang)

    if lowercase:
        text = text.lower()

    text = normalizer.normalize(text)
    tokens = tokenizer.tokenize(
        text,
        escape=False,
        return_str=True,
        protected_patterns=protected_patterns
    )
    return tokens.strip()

def _cealn_fn_(
        text: str,
        lang: str = "en",
        lowercase: bool = True,
        protected_patterns: Union[List[str], Tuple[str]] = PROTECTED_PATTENRS
) -> str:
    text = remove_non_utf8(text)
    text = remove_double_spaces(text)
    text = text.replace("@ - @", "-")

    text = moses_clean_sentences(
        text,
        lang=lang,
        lowercase=lowercase,
        protected_patterns=protected_patterns
    )

    return text

def get_clean_fn(
        input_unwrap_fn=lambda x: x,
        output_wrap_fn=lambda text, data: text
):
    """
    wrapper for clean function
    """

    def clean_fn(
            data: Any,
            lang: str = "en",
            lowercase: bool = True,
            protected_patterns: Union[List[str], Tuple[str]] = PROTECTED_PATTENRS
    ) -> str:
        text = input_unwrap_fn(data)
        text = _cealn_fn_(
            text,
            lang=lang,
            lowercase=lowercase,
            protected_patterns=protected_patterns
        )
        return output_wrap_fn(text, data)

    return clean_fn

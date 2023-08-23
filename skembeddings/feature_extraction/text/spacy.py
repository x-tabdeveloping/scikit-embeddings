from typing import Any, Iterable, Optional, Union

from sklearn.feature_extraction.text import CountVectorizer
from spacy.language import Language

from skembeddings.tokenization.spacy import SpacyTokenizer


class SpacyVectorizer(CountVectorizer):
    def __init__(
        self,
        model: Union[str, Language] = "en_core_web_sm",
        patterns: Optional[list[list[dict[str, Any]]]] = None,
        out_attrs: Iterable[str] = ("NORM",),
        ngram_range: tuple[int, int] = (1, 1),
        max_df: float = 1.0,
        min_df=1,
        max_features: Optional[int] = None,
    ):
        self.tokenizer = SpacyTokenizer(
            model, patterns=patterns, out_attrs=out_attrs
        )
        super(SpacyVectorizer, self).__init__(
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            analyzer=self.tokenizer.transform,  # type: ignore
        )

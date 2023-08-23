from typing import Optional

from sklearn.feature_extraction.text import CountVectorizer

from skembeddings.tokenization.huggingface import HuggingFaceTokenizer


class HuggingFaceVectorizer(CountVectorizer):
    def __init__(
        self,
        model: str = "bert-base-uncased",
        ngram_range: tuple[int, int] = (1, 1),
        max_df: float = 1.0,
        min_df=1,
        max_features: Optional[int] = None,
    ):
        self.tokenizer = HuggingFaceTokenizer(model)
        super(HuggingFaceVectorizer, self).__init__(
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            analyzer=self.tokenizer.transform,  # type: ignore
        )

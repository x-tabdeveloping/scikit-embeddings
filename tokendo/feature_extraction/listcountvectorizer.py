from collections import Counter
from typing import Iterable, TypeVar

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer

Elem = TypeVar("Elem")


class ListCountVectorizer(TransformerMixin, BaseEstimator):
    def __init__(self, sparse: bool = True):
        self.dict_vectorizer = DictVectorizer(sparse=sparse, sort=True)
        self.pandas_out = False

    def count_items(
        self, X: Iterable[Iterable[Elem]]
    ) -> Iterable[dict[Elem, int]]:
        for sub in X:
            yield Counter(sub)

    def fit(self, X: Iterable[Iterable], y=None):
        """Learns potential classes."""
        self.dict_vectorizer.fit(self.count_items(X))
        return self

    def transform(self, X: Iterable[Iterable], y=None):
        X_new = self.dict_vectorizer.transform(self.count_items(X))
        if not self.pandas_out:
            return X_new
        else:
            import pandas as pd

            return pd.DataFrame(X_new, columns=self.get_feature_names_out())  # type: ignore

    def get_feature_names_out(self):
        return self.dict_vectorizer.get_feature_names_out()

    def set_output(self, transform=None):
        """You can set the output of the pipeline to be a pandas dataframe.
        If you pass 'pandas' it will do this, otherwise it will disable pandas output.
        """
        if transform == "pandas":
            self.pandas_out = True
        else:
            self.pandas_out = False
        return self

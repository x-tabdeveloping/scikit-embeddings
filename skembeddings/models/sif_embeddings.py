from collections import Counter
from pathlib import Path
from typing import Iterable, Literal, Union

import numpy as np
from gensim import downloader
from gensim.models import KeyedVectors
from gensim.utils import SaveLoad
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer

from skembeddings.streams.utils import deeplist


class SIFEmbedding(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        word_vectors: Union[str, KeyedVectors] = "word2vec-google-news-300",
        smoothing: float = 1.0,
    ):
        self.word_vectors = word_vectors
        self.smoothing = smoothing
        if isinstance(word_vectors, str):
            if word_vectors in downloader.info()["models"]:
                self.keyed_vectors: KeyedVectors = downloader.load(word_vectors)  # type: ignore
            else:
                loaded_object = SaveLoad().load(self.word_vectors)
                if isinstance(loaded_object, KeyedVectors):
                    self.keyed_vectors = loaded_object
                else:
                    raise TypeError(
                        "Object loaded from disk is not a KeyedVectors instance."
                    )
        elif isinstance(word_vectors, KeyedVectors):
            self.keyed_vectors: KeyedVectors = word_vectors
        else:
            raise TypeError(
                f"You should pass a word_vectors name or keyed vectors to SIFEmbedding, not {type(word_vectors)}"
            )

    def _get_embedding(self, sent: list[str]) -> np.ndarray:
        embeddings = []
        for token in sent:
            try:
                emb = self.keyed_vectors[token]
                embeddings.append(emb)
            except KeyError:
                embeddings.append(np.full(self.n_features_out, np.nan))
        if not embeddings:
            return np.full(self.n_features_out, np.nan)
        embeddings = np.stack(embeddings)
        freqs = np.array([self.freq_[token] / self.total_ for token in sent])
        weighted: np.ndarray = (
            embeddings.T * (self.smoothing / (self.smoothing + freqs))
        ).T
        return np.sum(weighted, axis=0) / len(sent)

    def fit_transform(self, X: Iterable[Iterable[str]], y=None):
        self.n_features_out = self.keyed_vectors.vector_size
        X_eval: list[list[str]] = deeplist(X)
        self.freq_ = Counter()
        for sent in X_eval:
            self.freq_.update(sent)
        sent_embeddings = []
        self.total_ = self.freq_.total()
        for sent in X_eval:
            sent_embeddings.append(self._get_embedding(sent))
        sent_embeddings = np.stack(sent_embeddings)
        sent_embeddings = SimpleImputer().fit_transform(sent_embeddings)
        self.svd_ = TruncatedSVD(1).fit(sent_embeddings)
        self.sing_vect_ = self.svd_.components_
        u = self.sing_vect_
        X_new = np.stack([sent - u.T @ u @ sent for sent in sent_embeddings])
        return X_new

    def fit(self, X: Iterable[Iterable[str]], y=None):
        self.fit_transform(X, y)
        return self

    def transform(self, X: Iterable[Iterable[str]]) -> np.ndarray:
        if self.sing_vect_ is None:
            raise NotFittedError("Please fit the model before transforming.")
        X_eval: list[list[str]] = deeplist(X)
        sent_embeddings = []
        for sent in X_eval:
            sent_embeddings.append(self._get_embedding(sent))
        sent_embeddings = np.stack(sent_embeddings)
        u = self.sing_vect_
        X_new = np.stack([sent - u.T @ u @ sent for sent in sent_embeddings])
        return X_new

import tempfile
from typing import Iterable, Literal

import numpy as np
from confection import Config, registry
from gensim.models import KeyedVectors
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from tqdm import tqdm

from skembeddings.base import Serializable
from skembeddings.models._glove_impl import GloVeModel
from skembeddings.streams.utils import deeplist


class GloVeEmbedding(BaseEstimator, TransformerMixin, Serializable):
    def __init__(
        self,
        n_components: int = 100,
        agg: Literal["mean", "max", "both"] = "mean",
        alpha: float = 0.75,
        window_size: int = 20,
        batch_size: int = 128,
        learning_rate: float = 1e-2,
        epochs: int = 1,
        random_state: int = 0,
    ):
        self.agg = agg
        self.n_components = n_components
        self.alpha = alpha
        self.random_state = random_state
        self.batch_size = batch_size
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model_ = None
        self.loss_: list[float] = []
        self.n_features_out = (
            self.n_components if agg != "both" else self.n_components * 2
        )

    def fit(self, X: Iterable[Iterable[str]], y=None):
        self.model_ = GloVeModel(
            vector_size=self.n_components,
            alpha=self.alpha,
            seed=self.random_state,
            window_size=self.window_size,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        )
        self.partial_fit(X)
        return self

    def partial_fit(self, X: Iterable[Iterable[str]], y=None):
        if self.model_ is None:
            return self.fit(X)
        docs = deeplist(X)
        self.model_.update_vocab(docs)
        self.model_.update_cooccurrences(docs)
        for _ in range(self.epochs):
            self.model_.update_weights()
        return self

    def _collect_vectors_single(self, tokens: list[str]) -> np.ndarray:
        embeddings = []
        for token in tokens:
            try:
                embeddings.append(self.model_[token])  # type: ignore
            except KeyError:
                continue
        if not embeddings:
            return np.full((1, self.n_features_out), np.nan)
        return np.stack(embeddings)

    def transform(self, X: Iterable[Iterable[str]], y=None):
        """Transforms the phrase text into a numeric
        representation using word embeddings."""
        if self.model_ is None:
            raise NotFittedError("Model has not been fitted yet.")
        X: list[list[str]] = deeplist(X)
        embeddings = np.empty((len(X), self.n_features_out))
        for i_doc, doc in enumerate(tqdm(X)):
            if not len(doc):
                embeddings[i_doc, :] = np.nan
            doc_vectors = self.model_[doc]
            if self.agg == "mean":
                embeddings[i_doc, :] = np.nanmean(doc_vectors, axis=0)
            elif self.agg == "max":
                embeddings[i_doc, :] = np.nanmax(doc_vectors, axis=0)
            elif self.agg == "both":
                mean_vector = np.nanmean(doc_vectors, axis=0)
                max_vector = np.nanmax(doc_vectors, axis=0)
                embeddings[i_doc, :] = np.concatenate(
                    (mean_vector, max_vector)
                )
        return embeddings

    def to_bytes(self) -> bytes:
        pass

    def from_bytes(self, data: bytes):
        pass

    @property
    def config(self) -> Config:
        return Config(
            {
                "embedding": {
                    "@models": "glove_embedding.v1",
                    **self.get_params(),
                }
            }
        )

    @classmethod
    def from_config(cls, config: Config) -> "GloVeEmbedding":
        resolved = registry.resolve(config)
        return resolved["embedding"]

from pathlib import Path
from typing import Iterable, Literal, Union

import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from skembeddings.streams.utils import deeplist


class Word2VecEmbedding(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components: int = 100,
        window: int = 5,
        algorithm: Literal["cbow", "sg"] = "cbow",
        agg: Literal["mean", "max", "both"] = "mean",
        epochs: int = 5,
        random_state: int = 0,
        negative: int = 5,
        ns_exponent: float = 0.75,
        cbow_agg: Literal["mean", "sum"] = "mean",
        sample: float = 0.001,
        hs: bool = False,
        batch_words: int = 10000,
        shrink_windows: bool = True,
        learning_rate: float = 0.025,
        min_learning_rate: float = 0.0001,
        n_jobs: int = 1,
    ):
        self.agg = agg
        self.n_components = n_components
        self.n_jobs = n_jobs
        self.window = window
        self.algorithm = algorithm
        self.model = None
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.negative = negative
        self.ns_exponent = ns_exponent
        self.cbow_agg = cbow_agg
        self.sample = sample
        self.hs = hs
        self.batch_words = batch_words
        self.shrink_windows = shrink_windows
        self.epochs = epochs
        self.model_ = None
        self.loss_: list[float] = []
        self.n_features_out = (
            self.n_components if agg != "both" else self.n_components * 2
        )

    def _init_model(self, sentences=None) -> Word2Vec:
        return Word2Vec(
            sentences=sentences,
            vector_size=self.n_components,
            min_count=0,
            alpha=self.learning_rate,
            window=self.window,
            sample=self.sample,
            seed=self.random_state,
            workers=self.n_jobs,
            min_alpha=self.min_learning_rate,
            sg=int(self.algorithm == "sg"),
            hs=int(self.hs),
            negative=self.negative,
            ns_exponent=self.ns_exponent,
            cbow_mean=int(self.cbow_agg == "mean"),
            epochs=self.epochs,
            trim_rule=None,
            batch_words=self.batch_words,
            compute_loss=True,
            shrink_windows=self.shrink_windows,
        )

    def fit(self, X: Iterable[Iterable[str]], y=None):
        self._check_inputs(X)
        X = deeplist(X)
        self.loss_ = []
        self.model_ = self._init_model(sentences=X)
        self.loss_.append(self.model_.get_latest_training_loss())
        return self

    def partial_fit(self, X: Iterable[Iterable[str]], y=None):
        self._check_inputs(X)
        X = deeplist(X)
        if self.model_ is None:
            self.fit(X, y)
        else:
            self.model_.build_vocab(X, update=True)
            self.model_.train(
                X,
                total_examples=self.model_.corpus_count,
                epochs=self.model_.epochs,
                comput_loss=True,
            )
            self.loss_.append(self.model_.get_latest_training_loss())
        return self

    def _check_inputs(self, X):
        options = ["mean", "max", "both"]
        if self.agg not in options:
            raise ValueError(
                f"The `agg` value must be in {options}. Got {self.agg}."
            )

    def _collect_vectors_single(self, tokens: list[str]) -> np.ndarray:
        embeddings = []
        for token in tokens:
            try:
                embeddings.append(self.model_.wv[token])  # type: ignore
            except KeyError:
                continue
        if not embeddings:
            return np.full(self.n_features_out, np.nan)
        return np.stack(embeddings)

    def transform(self, X: Iterable[Iterable[str]], y=None):
        """Transforms the phrase text into a numeric
        representation using word embeddings."""
        self._check_inputs(X)
        X: list[list[str]] = deeplist(X)
        embeddings = np.empty((len(X), self.n_features_out))
        for i_doc, doc in enumerate(X):
            if not len(doc):
                embeddings[i_doc, :] = np.nan
            doc_vectors = self._collect_vectors_single(doc)
            if self.agg == "mean":
                embeddings[i_doc, :] = np.mean(doc_vectors, axis=0)
            elif self.agg == "max":
                embeddings[i_doc, :] = np.max(doc_vectors, axis=0)
            elif self.agg == "both":
                mean_vector = np.mean(doc_vectors, axis=0)
                max_vector = np.max(doc_vectors, axis=0)
                embeddings[i_doc, :] = np.concatenate(
                    (mean_vector, max_vector)
                )
        return embeddings

    @property
    def keyed_vectors(self) -> KeyedVectors:
        if self.model_ is None:
            raise NotFittedError(
                "Can't access keyed vectors, model has not been fitted yet."
            )
        return self.model_.wv

    @classmethod
    def from_pretrained(
        cls,
        model: Union[str, Path, Word2Vec],
        agg: Literal["mean", "max", "both"] = "mean",
    ):
        if isinstance(model, (str, Path)):
            model_ = Word2Vec.load(model)
        elif isinstance(model, Word2Vec):
            model_ = model
        else:
            raise TypeError(
                "Pretrained model either has to be a"
                "path or a Word2Vec instance."
            )
        res = cls(
            n_components=model_.vector_size,
            learning_rate=model_.alpha,
            window=model_.window,
            sample=model_.sample,
            random_state=model_.seed,
            n_jobs=model_.workers,
            min_learning_rate=model_.min_alpha,
            algorithm="sg" if model_.sg else "cbow",
            hs=bool(model_.hs),
            negative=model_.negative,
            ns_exponent=model_.ns_exponent,
            cbow_agg="mean" if model_.cbow_mean else "sum",
            epochs=model_.epochs if model_.epochs is not None else 5,
            batch_words=model_.batch_words,
            shrink_windows=model_.shrink_windows,
            agg=agg,
        )
        res.model_ = model_
        return res

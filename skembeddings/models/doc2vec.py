import io
import tarfile
from pathlib import Path
from typing import Iterable, Literal, Union

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import murmurhash3_32

from skembeddings.base import Serializable
from skembeddings.streams.utils import deeplist


def _tag_enumerate(docs: Iterable[list[str]]) -> list[TaggedDocument]:
    """Tags documents with their integer positions."""
    return [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]


class ParagraphEmbedding(BaseEstimator, TransformerMixin, Serializable):
    """Scikit-learn compatible Doc2Vec model."""

    def __init__(
        self,
        n_components: int = 100,
        window: int = 5,
        algorithm: Literal["dm", "dbow"] = "dm",
        tagging_scheme: Literal["hash", "closest"] = "hash",
        max_docs: int = 100_000,
        epochs: int = 10,
        random_state: int = 0,
        negative: int = 5,
        ns_exponent: float = 0.75,
        dm_agg: Literal["mean", "sum", "concat"] = "mean",
        dm_tag_count: int = 1,
        dbow_words: bool = False,
        sample: float = 0.001,
        hs: bool = False,
        batch_words: int = 10000,
        shrink_windows: bool = True,
        learning_rate: float = 0.025,
        min_learning_rate: float = 0.0001,
        n_jobs: int = 1,
    ):
        self.model_ = None
        self.loss_: list[float] = []
        self.seen_docs_ = 0
        if tagging_scheme not in ["hash", "closest"]:
            raise ValueError(
                "Tagging scheme should either be 'hash' or 'closest'"
            )
        self.algorithm = algorithm
        self.max_docs = max_docs
        self.n_components = n_components
        self.n_jobs = n_jobs
        self.window = window
        self.tagging_scheme = tagging_scheme
        self.epochs = epochs
        self.random_state = random_state
        self.negative = negative
        self.ns_exponent = ns_exponent
        self.dm_agg = dm_agg
        self.dm_tag_count = dm_tag_count
        self.dbow_words = dbow_words
        self.sample = sample
        self.hs = hs
        self.batch_words = batch_words
        self.shrink_windows = shrink_windows
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate

    def _tag_documents(
        self, documents: list[list[str]]
    ) -> list[TaggedDocument]:
        if self.model_ is None:
            raise TypeError(
                "You should not call _tag_documents"
                "before model is initialised."
            )
        res = []
        for document in documents:
            # While we have available slots we just add new documents to those
            if self.seen_docs_ < self.max_docs:
                res.append(TaggedDocument(document, [self.seen_docs_]))
            else:
                # If we run out, we choose a tag based on a scheme
                if self.tagging_scheme == "hash":
                    # Here we use murmur hash
                    hash = murmurhash3_32("".join(document))
                    id = hash % self.max_docs
                    res.append(TaggedDocument(document, [id]))
                elif self.tagging_scheme == "closest":
                    # We obtain the key of the most semantically
                    # similar document and use that.
                    doc_vector = self.model_.infer_vector(document)
                    key, _ = self.model_.dv.similar_by_key(doc_vector, topn=1)[
                        0
                    ]
                    res.append(TaggedDocument(document, [key]))
                else:
                    raise ValueError(
                        "Tagging scheme should either be 'hash' or 'closest'"
                        f" but {self.tagging_scheme} was provided."
                    )
            self.seen_docs_ += 1
        return res

    def _init_model(self, docs=None) -> Doc2Vec:
        return Doc2Vec(
            documents=docs,
            vector_size=self.n_components,
            min_count=0,
            alpha=self.learning_rate,
            window=self.window,
            sample=self.sample,
            seed=self.random_state,
            workers=self.n_jobs,
            min_alpha=self.min_learning_rate,
            dm=int(self.algorithm == "dm"),
            dm_mean=int(self.dm_agg == "mean"),
            dm_concat=int(self.dm_agg == "concat"),
            dbow_words=int(self.dbow_words),
            dm_tag_count=self.dm_tag_count,
            hs=int(self.hs),
            negative=self.negative,
            ns_exponent=self.ns_exponent,
            epochs=self.epochs,
            trim_rule=None,
            batch_words=self.batch_words,
            compute_loss=True,
            shrink_windows=self.shrink_windows,
        )

    def _append_loss(self):
        self.loss_.append(self.model_.get_latest_training_loss())  # type: ignore

    def fit(self, X: Iterable[Iterable[str]], y=None):
        """Fits a new doc2vec model to the given documents."""
        self.seen_docs_ = 0
        # Forcing evaluation
        X_eval: list[list[str]] = deeplist(X)
        n_docs = len(X_eval)
        if self.max_docs < n_docs:
            init_batch = _tag_enumerate(X_eval[: self.max_docs])
            self.model_ = self._init_model(init_batch)
            self._append_loss()
            self.partial_fit(X_eval[self.max_docs :])
            return self
        docs = _tag_enumerate(X_eval)
        self.model_ = self._init_model(docs)
        self._append_loss()
        return self

    def partial_fit(self, X: Iterable[Iterable[str]], y=None):
        """Partially fits doc2vec model (online fitting)."""
        # Force evaluation on iterable
        X_eval: list[list[str]] = deeplist(X)
        if self.model_ is None:
            self.fit(X_eval)
            return self
        # We obtained tagged documents
        tagged_docs = self._tag_documents(X_eval)
        # Then build vocabulary
        self.model_.build_vocab(tagged_docs, update=True)
        self.model_.train(
            tagged_docs,
            total_examples=self.model_.corpus_count,
            epochs=1,
            compute_loss=True,
        )
        self._append_loss()
        return self

    def transform(self, X: Iterable[Iterable[str]]) -> np.ndarray:
        """Infers vectors for all of the given documents."""
        if self.model_ is None:
            raise NotFittedError(
                "Model ha been not fitted, please fit before inference."
            )
        vectors = [self.model_.infer_vector(list(doc)) for doc in X]
        return np.stack(vectors)

    @property
    def components_(self) -> np.ndarray:
        if self.model_ is None:
            raise NotFittedError("Model has not been fitted yet.")
        return np.array(self.model_.dv.vectors).T

    def to_bytes(self) -> bytes:
        if self.model_ is None:
            raise NotFittedError(
                "Can't save model if it hasn't been fitted yet."
            )
        out_buffer = io.BytesIO()
        with tarfile.open(fileobj=out_buffer, mode="w:gz") as out_ball:
            model_file = io.BytesIO()
            self.model_.save(model_file)
            out_ball.addfile(tarfile.TarInfo("doc2vec.model"), model_file)
            return out_buffer.read()

    @classmethod
    def from_bytes(cls, data: bytes) -> "ParagraphEmbedding":
        in_buffer = io.BytesIO(data)
        with tarfile.open(fileobj=in_buffer, "r:gz") as in_ball:
            names = set(in_ball.getnames())
            if names != {"doc2vec.model"}:
                raise TypeError(
                    "Given path does not contain a serialized Doc2Vec model."
                )
            model_file = in_ball.extractfile("doc2vec.model")
            model = Doc2Vec.load(model_file)
            return cls.from_pretrained(model)


    @classmethod
    def from_pretrained(
        cls,
        model: Union[str, Path, Doc2Vec],
    ):
        if isinstance(model, (str, Path)):
            model_ = Doc2Vec.load(model)
        elif isinstance(model, Doc2Vec):
            model_ = model
        else:
            raise TypeError(
                "Pretrained model either has to be a"
                "path or a Doc2Vec instance."
            )
        if model_.dm_mean:
            dm_agg = "mean"
        elif model_.dm_concat:
            dm_agg = "concat"
        else:
            dm_agg = "sum"
        res = cls(
            n_components=model_.vector_size,
            learning_rate=model_.alpha,
            window=model_.window,
            sample=model_.sample,
            random_state=model_.seed,
            n_jobs=model_.workers,
            min_learning_rate=model_.min_alpha,
            algorithm="dm" if model_.dm else "dbow",
            dm_agg=dm_agg,
            dbow_words=bool(model_.dbow_words),
            dm_tag_count=model_.dm_tag_count,
            hs=bool(model_.hs),
            negative=model_.negative,
            ns_exponent=model_.ns_exponent,
            epochs=model_.epochs,
            batch_words=model_.batch_words,
            shrink_windows=model_.shrink_windows,
        )
        res.model_ = model_
        return res

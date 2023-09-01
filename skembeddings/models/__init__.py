from typing import Literal

from confection import registry

from skembeddings.error import NotInstalled

try:
    from skembeddings.models.word2vec import Word2VecEmbedding
except ModuleNotFoundError:
    Word2VecEmbedding = NotInstalled("Word2VecEmbedding", "gensim")

try:
    from skembeddings.models.doc2vec import ParagraphEmbedding
except ModuleNotFoundError:
    ParagraphEmbedding = NotInstalled("ParagraphEmbedding", "gensim")


@registry.models.register("word2vec_embedding.v1")
def make_word2vec_embedding(
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
    return Word2VecEmbedding(
        n_components=n_components,
        window=window,
        algorithm=algorithm,
        agg=agg,
        epochs=epochs,
        random_state=random_state,
        negative=negative,
        ns_exponent=ns_exponent,
        cbow_agg=cbow_agg,
        sample=sample,
        hs=hs,
        batch_words=batch_words,
        shrink_windows=shrink_windows,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        n_jobs=n_jobs,
    )


@registry.models.register("paragraph_embedding.v1")
def make_paragraph_embedding(
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
    return ParagraphEmbedding(
        n_components=n_components,
        window=window,
        algorithm=algorithm,
        tagging_scheme=tagging_scheme,
        max_docs=max_docs,
        epochs=epochs,
        random_state=random_state,
        negative=negative,
        ns_exponent=ns_exponent,
        dm_agg=dm_agg,
        dm_tag_count=dm_tag_count,
        dbow_words=dbow_words,
        sample=sample,
        hs=hs,
        batch_words=batch_words,
        shrink_windows=shrink_windows,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        n_jobs=n_jobs,
    )


__all__ = ["Word2VecEmbedding", "ParagraphEmbedding"]

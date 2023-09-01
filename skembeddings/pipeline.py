from pathlib import Path
from typing import Union

from confection import Config, registry
from sklearn.pipeline import Pipeline

from skembeddings.base import Serializable


class EmbeddingPipeline(Pipeline):
    def __init__(
        self,
        tokenizer: Serializable,
        model: Serializable,
        frozen: bool = False,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.frozen = frozen
        steps = [("tokenizer_model", tokenizer), ("embedding_model", model)]
        super().__init__(steps=steps)

    def freeze(self):
        self.frozen = True
        return self

    def unfreeze(self):
        self.frozen = False
        return self

    def fit(self, X, y=None, **kwargs):
        if self.frozen:
            return self
        super().fit(X, y=y, **kwargs)

    def partial_fit(self, X, y=None, classes=None, **kwargs):
        """
        Fits the components, but allow for batches.
        """
        if self.frozen:
            return self
        for name, step in self.steps:
            if not hasattr(step, "partial_fit"):
                raise ValueError(
                    f"Step {name} is a {step} which does"
                    "not have `.partial_fit` implemented."
                )
        for name, step in self.steps:
            if hasattr(step, "predict"):
                step.partial_fit(X, y, classes=classes, **kwargs)
            else:
                step.partial_fit(X, y)
            if hasattr(step, "transform"):
                X = step.transform(X)
        return self

    @property
    def config(self) -> Config:
        embedding: Serializable = self["embedding_model"]  # type: ignore
        tokenizer: Serializable = self["tokenizer_model"]  # type: ignore
        return tokenizer.config.merge(embedding.config)

    def to_disk(self, path: Union[str, Path]) -> None:
        embedding: Serializable = self["embedding_model"]  # type: ignore
        tokenizer: Serializable = self["tokenizer_model"]  # type: ignore
        path = Path(path)
        path.mkdir(exist_ok=True)
        config_path = path.joinpath("config.cfg")
        tokenizer_path = path.joinpath("tokenizer.bin")
        embedding_path = path.joinpath("embedding.bin")
        with open(embedding_path, "wb") as embedding_file:
            embedding_file.write(embedding.to_bytes())
        with open(tokenizer_path, "wb") as tokenizer_file:
            tokenizer_file.write(tokenizer.to_bytes())
        self.config.to_disk(config_path)

    @classmethod
    def from_disk(cls, path: Union[str, Path]) -> "EmbeddingPipeline":
        path = Path(path)
        config_path = path.joinpath("config.cfg")
        tokenizer_path = path.joinpath("tokenizer.bin")
        embedding_path = path.joinpath("embedding.bin")
        config = Config().from_disk(config_path)
        resolved = registry.resolve(config)
        with open(tokenizer_path, "rb") as tokenizer_file:
            tokenizer = resolved["tokenizer"].from_bytes(tokenizer_file.read())
        with open(embedding_path, "rb") as embedding_file:
            embedding = resolved["embedding"].from_bytes(embedding_file.read())
        return cls(tokenizer, embedding)

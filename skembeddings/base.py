from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Union

import joblib
from confection import Config


class Serializable(ABC):
    def to_bytes(self) -> bytes:
        buffer = BytesIO()
        joblib.dump(self, buffer)
        return buffer.read()

    def from_bytes(self, data: bytes):
        buffer = BytesIO(data)
        return joblib.load(buffer)

    @property
    def config(self) -> Config:
        return Config()

    @abstractmethod
    def from_config(cls, config: Config) -> "Serializable":
        pass

    def to_disk(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(exist_ok=True)
        model_path = path.joinpath("model.bin")
        with open(model_path, "wb") as model_file:
            model_file.write(self.to_bytes())
        config_path = path.joinpath("config.cfg")
        self.config.to_disk(config_path)

    @classmethod
    def from_disk(cls, path: Union[str, Path]):
        path = Path(path)
        model_path = path.joinpath("model.bin")
        config_path = path.joinpath("config.cfg")
        with open(model_path, "rb") as model_file:
            model_data = model_file.read()
        config = Config().from_disk(config_path)
        res = cls.from_config(config)
        return res.from_bytes(model_data)

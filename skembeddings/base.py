from abc import ABC
from io import BytesIO
from pathlib import Path
from typing import Union

import joblib


class Serializable(ABC):
    def to_bytes(self) -> bytes:
        buffer = BytesIO()
        joblib.dump(self, buffer)
        return buffer.read()

    @classmethod
    def from_bytes(cls, data: bytes):
        buffer = BytesIO(data)
        return joblib.load(buffer)

    def to_disk(self, path: Union[str, Path]) -> None:
        with open(path, "wb") as out_file:
            out_file.write(self.to_bytes())

    @classmethod
    def from_disk(cls, path: Union[str, Path]):
        with open(path, "rb") as in_file:
            buffer = in_file.read()
            return cls.from_bytes(buffer)

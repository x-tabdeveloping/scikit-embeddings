from pathlib import Path
from typing import Union

import joblib


class SaveLoad:
    def save(self, path: Union[str, Path]) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Union[str, Path]):
        return joblib.load(path)

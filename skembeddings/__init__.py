import catalogue
from confection import Config, registry

from skembeddings import models, tokenizers

registry.tokenizers = catalogue.create(
    "confection", "tokenizers", entry_points=False
)
registry.models = catalogue.create("confection", "models", entry_points=False)

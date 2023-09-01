import catalogue
from confection import Config, registry

registry.tokenizers = catalogue.create(
    "confection", "tokenizers", entry_points=False
)
registry.models = catalogue.create("confection", "models", entry_points=False)

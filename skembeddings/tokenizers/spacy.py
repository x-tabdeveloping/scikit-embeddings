from typing import Any, Iterable, Optional, Union

import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Doc, Token

from skembeddings.base import SaveLoad

# We create a new extension on tokens.
if not Token.has_extension("filter_pass"):
    Token.set_extension("filter_pass", default=False)

ATTRIBUTES = {
    "ORTH": "orth_",
    "NORM": "norm_",
    "LEMMA": "lemma_",
    "UPOS": "pos_",
    "TAG": "tag_",
    "DEP": "dep_",
    "LOWER": "lower_",
    "SHAPE": "shape_",
    "ENT_TYPE": "ent_type_",
}


class SpacyTokenizer(BaseEstimator, TransformerMixin, SaveLoad):
    def __init__(
        self,
        model: Union[str, Language] = "en_core_web_sm",
        patterns: Optional[list[list[dict[str, Any]]]] = None,
        out_attrs: Iterable[str] = ("NORM",),
    ):
        self.model = model
        if isinstance(model, Language):
            self.nlp = model
        elif isinstance(model, str):
            self.nlp = spacy.load(model)
        else:
            raise TypeError(
                "'model' either has to be a spaCy"
                "nlp object or the name of a model."
            )
        self.patterns = patterns
        self.out_attrs = tuple(out_attrs)
        for attr in self.out_attrs:
            if attr not in ATTRIBUTES:
                raise ValueError(f"{attr} is not a valid out attribute.")
        self.matcher = Matcher(self.nlp.vocab)
        self.matcher.add(
            "FILTER_PASS",
            patterns=[] if self.patterns is None else self.patterns,
        )

    def fit(self, X, y=None):
        """Exists for compatiblity, doesn't do anything."""
        return self

    def partial_fit(self, X, y=None):
        """Exists for compatiblity, doesn't do anything."""
        return self

    def label_matching_tokens(self, docs: list[Doc]):
        """Labels tokens that match one of the given patterns."""
        for doc in docs:
            if self.patterns is not None:
                matches = self.matcher(doc)
            else:
                matches = [(None, 0, len(doc))]
            for _, start, end in matches:
                for token in doc[start:end]:
                    token._.set("filter_pass", True)

    def token_to_str(self, token: Token) -> str:
        """Returns textual representation of token."""
        attributes = [
            getattr(token, ATTRIBUTES[attr]) for attr in self.out_attrs
        ]
        return "|".join(attributes)

    def transform(self, X: Iterable[str]) -> list[list[str]]:
        if isinstance(X, str):
            raise TypeError(
                "str passed instead of iterable, did you mean to pass [X]?"
            )
        docs = list(self.nlp.pipe(X))
        # Label all tokens according to the patterns.
        self.label_matching_tokens(docs)
        res: list[list[str]] = []
        for doc in docs:
            tokens = [
                self.token_to_str(token)
                for token in doc
                if token._.filter_pass
            ]
            res.append(tokens)
        return res

    def get_feature_names_out(self, input_features=None):
        return None

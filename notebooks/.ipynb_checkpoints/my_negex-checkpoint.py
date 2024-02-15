from dataclasses import dataclass
from pydoc import resolve
from tabnanny import verbose
#from tkinter import BEVEL
import spacy
from spacy.tokens import Span, Token, Doc
from spacy.matcher import Matcher, DependencyMatcher
from spacy.util import filter_spans
import typing

from negspacy.negation import Negex
from spacy.language import Language
from negspacy.termsets import termset

default_ts = termset("en_clinical").get_patterns()

@Language.factory(
        "my_negex",
        default_config={
        "neg_termset": default_ts,
        "ent_types": list(),
        "extension_name": "my_negex",
        "chunk_prefix": list(),
    },
)



class MyNegex:
        def __init__(
                self,
                nlp:Language,
                name: str,
                neg_termset: dict,
                ent_types: list,
                extension_name: str,
                chunk_prefix: list,
        ):
                self.name = name
                self.extension_name = extension_name
                self.wrapped = Negex(nlp,"negex",neg_termset,ent_types,extension_name,chunk_prefix)
        
        def my_negex(self,doc):
                n = self.wrapped
                preceding, following, terminating = n.process_negations(doc)
                boundaries = n.termination_boundaries(doc, terminating)
                for b in boundaries:
                        sub_preceding = [i for i in preceding if b[0] <= i[1] < b[1]]
                        sub_following = [i for i in following if b[0] <= i[1] < b[1]]
                
                # THIS IS THE SINGLE CHANGE!!!!
                        for nc in doc[b[0] : b[1]].noun_chunks:
                                
                                if any(pre < nc.start for pre in [i[1] for i in sub_preceding]):
                                        nc._.set(self.extension_name, True)
                                        continue
                                if any(fol > nc.end for fol in [i[2] for i in sub_following]):
                                        nc._.set(self.extension_name, True)
                                        continue
                                if n.chunk_prefix:
                                        if any(
                                                nc.text.lower().startswith(c.text.lower())
                                                for c in n.chunk_prefix
                                        ):
                                                nc._.set(self.extension_name, True)
                return doc
        
        
        def __call__(self,doc):
                return self.my_negex(doc)


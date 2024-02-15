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
import re
import coreferee



from my_negex import MyNegex

Doc.set_extension("beliefs", default=[], force=True)
Span.set_extension("beliefs", default=[], force=True)

mental_state_verbs = "believe, think, doubt, fear, feel, think, know, understand, imagine, realize, recognize, mean, remember, forget, suppose, want, intend, try, aim, need, prefer".split(", ")
mental_state_verbs += "like, love, hate, dislike, want, hope, wish, feel, need, prefer, enjoy, appreciate, fear, envy, care, mind".split(", ")
mental_state_verbs += "agree, disagree".split(", ")
mental_state_verbs = set(mental_state_verbs)

class VerbChunk:
    def __init__(
        self,
        verb_span: Span,
        # Object side refers to the token in the span that should be used as the 
        # parent of the subject
        subject_side: Token,
        # Object side refers to the token in the span that should be used as the 
        # parent of the object
        object_side: Token
    ):
        self.verb_span = verb_span
        self.object_side = object_side
        self._subject_side = subject_side
        self.attributional = is_mental_state_verb(self.subject_side)
    
    @property
    def subject_side(self):
        return self._subject_side

    @subject_side.setter
    def subject_side(self,new_verb):
        self._subject_side = new_verb
        self.attributional = is_mental_state_verb(self._subject_side)

    def __repr__(self):
        return f"{self.verb_span} - Subject Root: {self.subject_side} Object Root: {self.object_side} Attributional: {self.attributional}"
    
    def to_dict(self):
        return {
            "verb_span":self.verb_span,
            "object_side":self.object_side,
            "subject_side":self._subject_side,
            "attributional":self.attributional
        }

@dataclass
class ProxyToken:
    text: str

class Belief:
    def __init__(
        self,
        subject: Span,
        verb: VerbChunk,
        object: Span,

    ):
        self.resolved_subject = None
        self.subject = subject
        self.verb = verb
        self.object = object
        self.nested_beliefs = []
        self.negated = False
        

    @property
    def subject(self):
        return self._subject
    
    @subject.setter
    def subject(self,nsubj):
        self._subject = nsubj
        self.resolved_subject = self.resolve_subject(nsubj)
       
    
    def resolve_subject(self,subj):
        result = []
        for ot in subj:

            if ot.pos_ in ("PRON"):
                toks = ot.doc._.coref_chains.resolve(ot)
                if toks:
                    result = [] 
                    for i,t in enumerate(toks):
                        if i>0:
                            result.append(ProxyToken("and"))
                        if ot.morph.to_dict().get("Poss")=="Yes":
                            result.append(ProxyToken(t.text+"'s"))
                        else:
                            result.append(t)
                elif ot.lemma_ == "who" and ot.doc[max(0,ot.i-1)].pos_=="PROPN":
                    result.append(ot.doc[ot.i-1])
                else: 
                    # We found a pronoun that can't be resolved, so bail
                    return None
            else:
                result.append(ot)
        return result
        

   
    def handle_negation(self):
        result = [t for t in self.verb.verb_span]
        if self.negated:
            #print(f"Want to negate {self.verb.verb_span.root} at {self.verb.verb_span.root.i}")
           
            auxpos = _find_pos(self.verb.verb_span,"AUX")
            if auxpos > -1:
                result.insert(auxpos+1,ProxyToken("not"))
            elif self.verb.verb_span.root.pos_ == 'VERB' and self.verb.verb_span.root.morph.to_dict().get("Tense") == "Past":
                pos = self.verb.verb_span.root.i - self.verb.verb_span.start
                del result[pos]
                result[pos:pos] = [ProxyToken("did"),ProxyToken("not"),ProxyToken(self.verb.verb_span.root.lemma_)]
                #print(f"Verb {v.text} lemma {v.lemma_} morph {v.morph}")
            else:
                result.insert(0,ProxyToken("not"))

        return result

    def smart_join(self,l,end = False):
        result = []
        
        for i,s in enumerate(l):
            if len(s.text.strip())==0:
                continue
            elif i==0 or (type(s)==Token and ((s.pos_=="PART" and not re.search('[aeiou]',s.text) or (s.pos_=="PUNCT" and s.text not in ["'",'"'])))):
                result.append(s.text)
            else:
                result.append(" ")
                result.append(s.text)
        if end and result[-1] not in ["!",".","?",'"',"'",")"]:
            result.append(".")
        return "".join(result)



    def as_string(self):
       
        tokens = [t for t in (self.resolved_subject if self.resolved_subject else self.subject)]
        tokens += [t for t in self.handle_negation()]
        tokens += [t for t in self.object] if self.object else []
        return self.smart_join(tokens).capitalize()


    def _generate_string(self,indent = ""):
        s =  f"{indent}Subj:{self.subject} : Verb:{self.verb.verb_span} : Object:{self.object}"
        if self.is_attributional():
            s = f"{s} [ATTRIBUTED]"
        if self.negated:
            s = f"{s} [NEGATED]"
        if self.nested_beliefs:
            for b in self.nested_beliefs:
                s = f"{s}-->\n{b._generate_string(f'{indent}    ')}"
        return s

    def __repr__(self):
        return self._generate_string()

    def mark_negated(self,b):
        self.negated = b

    def is_attributional(self):
        return self.verb.attributional

    def is_self_belief(self):
        return not self.nested_beliefs

    def clean_subject(self):
        tokens = [tok for tok in self.subject]
        result = []

        #TODO: Refactor this with resolve_subject
        for ot in tokens:

            if ot.pos_ in ("PRON"):
                toks = ot.doc._.coref_chains.resolve(ot)
                if toks:
                    result = [] 
                    for i,t in enumerate(toks):
                        if i>0:
                            result.append(ProxyToken("and"))
                        if ot.morph.to_dict().get("Poss")=="Yes":
                            result.append(ProxyToken(t.text+"'s"))
                        else:
                            result.append(t)
                elif ot.lemma_ == "who" and ot.doc[max(0,ot.i-1)].pos_=="PROPN":
                    result.append(ot.doc[ot.i-1])
                else: 
                    result.append(ot)
            else:
                result.append(ot)
        
        return " ".join([t.text for t in result])


    def clean_belief(self):
        result = []
        #print(f"Processing '{self.subject}' is attributional {self.is_attributional()}")
        
        # take care of deictic references
        det = _find_matching_child(self.subject.root,["det"])
       
        if det and det.root.morph.to_dict().get("PronType") == "Dem" and det.root.morph.to_dict().get("Number")=="Sing":
            #print("Excluding because of deixis")
            result = []
        
        elif self.is_attributional():
            #print(f"Processing attributional layer with person {self.subject.root.morph.to_dict().get('Person')}")
            if self.subject.root.morph.to_dict().get("Person") == '1':
                result = [sb.clean_belief() for sb in self.nested_beliefs]
            else:
                result = [(self.as_string(),self.clean_subject())]    
         
        elif self.subject.root.lemma_ in ["it","there"]:
            #print("Excluding because of it / there")
            result = [sb.clean_belief() for sb in self.nested_beliefs]
        
        # Remove pronouns
        elif self.subject.root.pos_ == "PRON" and not self.resolved_subject and not self.subject.root.lemma_=="we":            
            result = []
        
        else:
            result += [(self.as_string(),self.clean_subject())]
        
        return flatten(result)
    
    def to_dict(self):
        return {
            "resolved_subject":self.resolved_subject,
            "subject":self.subject,
            "verb":self.verb.to_dict(),
            "nested_beliefs":[x.to_dict() for x in self.nested_beliefs],
            "negated":self.negated
        }


def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])

def _find_pos(span,pos):
    if not isinstance(pos,list):
        pos = [pos]
    for i,x in enumerate(span):
        if x.pos_ in pos:
            return i
    return -1
       

def _get_verb_matches(span):
    # 1. Find verb phrases in the span
    # (see mdmjsh answer here: https://stackoverflow.com/questions/47856247/extract-verb-phrases-using-spacy)
    match_keys = {
        1:"deep_match",
        2:"aux_verb_phrase",
        3:"idiomatic_wants",
        4:"plain_verb",
        5:"conj_verb",
        6:"xing_to_y"}

    deep_matcher = DependencyMatcher(span.vocab)
    pattern = [
    {
        "RIGHT_ID": "active_verb",
        "RIGHT_ATTRS": {"POS": {"IN":["VERB","AUX"]}}
    },
    {
        "LEFT_ID": "active_verb",
        "REL_OP": ">",
        "RIGHT_ID": "operating_aux",
        "RIGHT_ATTRS": {"POS": "AUX","DEP":{"NOT_IN":["ccomp","conj"]}}
    }]
    deep_matcher.add(1,[pattern])

    pattern = [
    {
        "RIGHT_ID": "active_verb",
        "RIGHT_ATTRS": {"POS": {"IN":["VERB","AUX"]}}
    },
    {
        "LEFT_ID": "active_verb",
        "REL_OP": ">",
        "RIGHT_ID": "comp_verb",
        "RIGHT_ATTRS": {"POS": "VERB","DEP":"xcomp"}
    },
    {
        "LEFT_ID": "comp_verb",
        "REL_OP": ">",
        "RIGHT_ID": "part",
        "RIGHT_ATTRS": {"POS": "PART"}
    }]
    deep_matcher.add(6,[pattern])

    # LL: Not sure if this helps or conflicts with the previous pattern
    # pattern = [
    #     {
    #         "RIGHT_ID": "subject_action",
    #         "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "VERB", 'AUX']}}
    #     },
    #     {
    #         "LEFT_ID": "subject_action",
    #         "REL_OP": ">",
    #         "RIGHT_ID": "is",
    #         "RIGHT_ATTRS": {"POS": {"IN": ["AUX", "VERB"]}, "DEP": "advcl"}
    #     },
    #     {
    #         "LEFT_ID": "subject_action",
    #         "REL_OP": ">",
    #         "RIGHT_ID": "suject",
    #         "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", 'ADJ']}, "DEP": "amod"}
    #     },
    # ]
    #
    # deep_matcher.add("long_subject_action", [pattern])

    matches = deep_matcher(span)
    matched_spans = [(match_keys[m[0]],span[min(m[1]):max(m[1])+1]) for m in matches]
    verb_matcher = Matcher(span.vocab)
    ## JEI: Adding more complex verbs
    verb_matcher.add(2, [
        [{"POS": "AUX", "OP":'+'}, {"POS": "VERB", "OP":"?"}]])

    # JEI: Matching constructions like <Person> would have <Person> <verb>
    verb_matcher.add(3, [[
        {"ORTH": "would"},
        {"ORTH":"have"},
        {"POS": {"IN":["PROPN","PRON"]}},
        {"POS": {"IN":["VERB","AUX"]}}]])

    verb_matcher.add(4, [[{"POS": "VERB"}]])

    verb_matcher.add(5, [[{"POS": {"IN":["VERB","AUX"]}},{"ORTH":"and"},{"POS": {"IN":["VERB","AUX"]}}]])

    matches = verb_matcher(span)
    matched_spans += [(match_keys[id],span[start:end]) for id, start, end in matches]

    return matched_spans

def _get_maximal_spans(spans):
    spans = sorted(spans,key=lambda x:x.start)
    merged = []
    for s in spans:
        if not merged:
            merged.append(s)
        elif merged[-1].end < s.start:
            merged.append(s)
        elif s.end > merged[-1].end:
            merged[-1] = Span(s.doc,merged[-1].start,s.end)
    return merged

def _get_verb_chunks(span) -> typing.List[VerbChunk]:
    matches = _get_verb_matches(span)
  
    # Filter matches (e.g. do not have both "has won" and "won" in verbs)
    verb_chunks = []

    # Screening out adverbial clauses
    filtered_spans = [x for x in _get_maximal_spans([m[1] for m in matches]) if x.root.dep_!="advcl"]
    
   


    for f in filtered_spans:
        last = None
        first = None
        for t in f:
            if t.pos_ in ["VERB","AUX"]:
                if first is None:
                    first = t
                last = t
        # For now, marking subject & object side based on start and end of the span
        # Might need to be smarter - this is one of those places where understanding
        # how the dependency tree is built would help!
        vs = Span(f[0].doc,f[0].i,f[-1].i+1)
        ss,os = first,last
        if ss.dep_ == "auxpass":
            ss = ss.head
        
        if vs.root.dep_ == "ccomp" and vs.root.head.i < vs.root.i:
           os = vs.root.head
        
        verb_chunks.append(VerbChunk(vs,ss,os))
    return verb_chunks
        

def _update_subject_side(verb_chunk):
    root = verb_chunk.subject_side
    
    while root:
        # Can we find subject at current level?
        for c in root.children:
            if c.dep_ in ["nsubj", "nsubjpass","expl"]:
                subject = extract_span_from_entity(c)
                verb_chunk.subject_side = root
                return subject

        # ... otherwise recurse up one level
        if (root.dep_ in ["conj", "cc", "advcl", "acl", "ccomp", "aux"]
            and root != root.head):
            root = root.head
        else:
            root = None

    return None

def is_mental_state_verb(verb: Token):
    return verb.lemma_ in mental_state_verbs

def _find_matching_child(root, allowed_types):
    for c in root.children:
        if c.dep_ in allowed_types:
            return extract_span_from_entity(c)
    return None

def spans_intersect(span_a,span_b):
    return (span_a.start >= span_b.start and span_a.start < span_b.end) or \
        (span_b.start >= span_a.start and span_b.start < span_a.end)


def extract_span_from_entity(token):
    ent_subtree = sorted([c for c in token.subtree], key=lambda x: x.i)
    return Span(token.doc, start=ent_subtree[0].i, end=ent_subtree[-1].i + 1)

def generate_belief_for_chunk(vc:VerbChunk) -> Belief:
  
    subject = _update_subject_side(vc)
    if not subject:
        return None
    object_tokens = set()
    for v in vc.object_side.rights:
        object_tokens.update([t for t in v.subtree])
        #if v.i >= vc.verb_span.end:
            
    
    tl = sorted(object_tokens,key=lambda x:x.i)
    result = Belief(subject,vc,Span(subject.doc,max(tl[0].i,vc.verb_span.end),tl[-1].i+1) if tl else None)
    return result
    

def extract_beliefs(span):
    #print(f"Processing {span}")
    verb_chunks = _get_verb_chunks(span)
   
    beliefs = [b for b in [generate_belief_for_chunk(vc) for vc in verb_chunks] if b is not None]
    beliefs = _parse_conjunctions(beliefs) 
    beliefmap = {}
    
    #TODO check if verb is nested in object - if so, remove
    for b in beliefs:
        if b.object:
            beliefmap[(b.subject.start,b.object.start,b.object.end)] = b
        else:
            beliefmap[(b.subject.start,b.verb.verb_span.start,b.verb.verb_span.end)] = b
    #print(beliefmap)
    noparent_children = [p for p in beliefmap.keys()]
  
    for child in list(noparent_children):
        bestparent = None
        for parent in beliefmap.keys():
            if parent == child:
                continue

            # Going to find the shortest parent that contains this child
            if child[2] <= parent[2] and (child[0] >= parent[1] or child[1] > parent[1]):
                if child in noparent_children:
                    noparent_children.remove(child)
                if bestparent is None or bestparent[2]-bestparent[1] > parent[2] - parent[1]:
                    bestparent = parent
        if bestparent:
            beliefmap[bestparent].nested_beliefs.append(beliefmap[child])
    return [beliefmap[c] for c in noparent_children]

def _spans_cconj(token):
    #print(f"Token span should be {token.head}:{token.head.i} to {token}:{token.i}")
    for i in range(token.head.i,token.i):
        if token.doc[i].dep_ == "CCONJ":
            return True
    return False


def _parse_conjunctions(beliefs):
    beliefmap = {b.verb.object_side:b for b in beliefs}
   
    for b in beliefs:
        if not b.object: 
            continue
        root_token = b.verb.verb_span.root

        if root_token.head in beliefmap and \
            root_token.dep_ == "conj" and _spans_cconj(root_token):
          
            parent_object_span = beliefmap[root_token.head].object
            start,end = parent_object_span.start,parent_object_span.end
            end = min(b.object.end,end)
            if (b.subject.start > start):
                end = b.subject.start
            else:
                end = min(b.verb.verb_span.start,b.object.start)
            
            # Back out any final conjunction or punctuation
            while end > start and b.object.doc[end-1].pos_ in ["CCONJ","PUNCT"]:
                end-=1
            beliefmap[root_token.head].object = Span(b.object.doc,start,end)
    
    return beliefs


@Language.component('beliefs')
def extract_beliefs_doc(doc):
    noun_chunks = [nc for nc in doc.noun_chunks]
    for sent in doc.sents:
        if sent[-1].text =="?" or sent[0].lemma_ in ['who','what','how','why','where','when']:
            continue
        
        try:
            beliefs = extract_beliefs(sent)
        except Exception as e:
            print(f"Error {e} parsing {sent.text}")
            beliefs = []
        
        queue = set(beliefs)
        while queue:
            b = queue.pop()
            if b.nested_beliefs:
                queue.update(b.nested_beliefs)
            for nc in noun_chunks:
                if spans_intersect(b.subject,nc) and nc._.my_negex:
                    b.mark_negated(True)
        sent._.beliefs = beliefs
        doc._.beliefs += sent._.beliefs
    return doc


def add_to_pipe(nlp):
    if "my_negex" not in nlp.pipe_names:
        nlp.add_pipe("my_negex")
    nlp.add_pipe('coreferee')
    nlp.add_pipe('beliefs')
   
    # coreferee works with spacy==3.1.6 coreferee==1.1.3 en-core-web-lg==3.1.0 coreferee-model-en==1.0.0
    # setting up:
    # On tiamat install local tensorflow
    # pip install --ignore-installed --upgrade tensorflow-2.5.0-cp39-cp39-linux_x86_64.whl
    # pip install --user spacy==3.1.6
    # pip install --user coreferee
    # python -m spacy download en_core_web_lg
    # python -m coreferee install en

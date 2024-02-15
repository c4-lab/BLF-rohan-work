import spacy
from spacy_langdetect import LanguageDetector
import neuralcoref
import re
import time
import ray
import os
import pandas as pd



files = os.listdir("CleanedData/")


processCount = len(files)
ray.init(num_cpus=processCount)

def generate_subject_predicate(verbs, subjects, objects):
    # create the combinations
    pairs = []
    verb_dict = {}
    for verb in verbs:
        verb_dict[verb.text] = {'subj': [], 'pred': []}
        for sub in subjects:
            if sub.root.head.text == verb.text:
                # sent_subj = sub.text
                verb_dict[verb.text]['subj'].append(sub.text)
        for obj in objects:
            try:
                if obj.head.text == verb.text:
                    objstring = obj.text
                    # sent_pred = verb.text + '-----' + obj.text
                    # pair = sent_subj + '-----' + sent_pred
                    # pairs.append (pair)
                    verb_dict[verb.text]['pred'].append(objstring)
                elif obj.head.pos_ in ['ADP']:
                    objstring = obj.head.text + ' ' + obj.text
                    verb_dict[verb.text]['pred'].append (objstring)
            except:
                if obj.root.head.text == verb.text:
                    objstring = obj.text
                    verb_dict[verb.text]['pred'].append (objstring)
    # pprint(verb_dict)
    for verb in verb_dict.keys():
        sent_subj = ' '.join(verb_dict[verb]['subj']) if len(verb_dict[verb]['subj'])>0 else 'EMPTY'
        sent_pred = verb + '-----' + (' '.join(verb_dict[verb]['pred']) if len(verb_dict[verb]['pred'])>0 else 'EMPTY')
        pair = sent_subj + '-----' + sent_pred
        if pair.count('EMPTY') > 0:
            continue
        pairs.append (pair)
        # print(pair)
    # pairs = '\t'.join (pairs)
    return pairs

def parse(sentence,nlp):
    
    pairs = ''
    doc = nlp (sentence)

    # identify the verbs
    verbs = []
    for token in doc:
        # print (token.text, token.dep_, token.pos_, token.head)
        if token.pos_ in ['VERB', 'AUX'] and token.dep_ == 'ROOT':
            verbs.append (token)
    if len (verbs) == 0:
        pairs = None
    # identify the subjects
    subjects = []
    for chunk in doc.noun_chunks:
        # print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text, chunk.root.head.dep_)
        if (chunk.root.dep_ in ['nsubj', 'nsubjpass'] and chunk.root.head.pos_ in ['VERB', 'AUX']) \
                or (chunk.root.dep_ in ['conj'] and chunk.root.head.dep_ in ['nsubj', 'nsubjpass']):
            subjects.append (chunk)
    # identify the objects
    objects = []
    for chunk in doc.noun_chunks:
        # print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text, chunk.root.head.dep_)
        if (chunk.root.dep_ in ['dobj', 'attr'] and chunk.root.head.pos_ in ['VERB', 'AUX']) \
                or (chunk.root.dep_ in ['conj'] and chunk.root.head.dep_ in ['dobj', 'pobj']):
            objects.append (chunk)
    # identify the complements
    for token in doc:
        # print (token.text, token.dep_, token.pos_, token.head)
        if (token.dep_ in ['acomp'] and token.head.pos_ in ['VERB', 'AUX']):
            # print(token)
            objects.append(token)
        elif (token.dep_ in ['pobj'] ):
            # and token.head.dep_ in ['agent']
            objects.append(token)
    # displacy.serve (doc, style="dep", host='127.0.0.1')
    pairs = generate_subject_predicate (verbs, subjects, objects)
    return pairs

@ray.remote
def prepExtract(fileName,writeFileName):
    print(fileName)
    df = pd.read_feather("CleanedData/"+fileName)
    wf = open(writeFileName,"w")
    nlp = spacy.load ('en_core_web_lg')
    nlp.add_pipe (LanguageDetector (), name="language_detector", last=True)
    neuralcoref.add_to_pipe (nlp)
    
    
    for i in range(len(df)):
        sentence = df.iloc[i]['cleanText']
        hashtag = df.iloc[i]['tweet_hash']
        if i%1000 == 0:
            print(fileName+" : "+str(i))
        pairs = parse(sentence,nlp)
        for j, pair in enumerate(pairs):
            pair_id = hashtag+'#'+str(j)+' : '+pair
            wf.write(pair_id)
            wf.write("\n")
    wf.close()

t = time.time()


for i in range(len(files)):
    prepExtract.remote(files[i],"preposition"+str(i)+".txt")



print(time.time()-t)
del results,df,result_ids
ray.shutdown()

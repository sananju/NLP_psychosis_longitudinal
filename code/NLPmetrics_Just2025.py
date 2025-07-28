"""
Code to compute different NLP metrics, as in Just et al., 2025:
DOI to follow.
Please cite the paper above if you use this code for your own work.
Authors: Galina Ryazanskaya and Sandra Just 25/05/2025
"""

import numpy as np
import torch
import spacy
import re
import networkx as nx

from typing import List
from wordfreq import word_frequency
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertForNextSentencePrediction

#Load German spacy model
nlp = spacy.load("de_core_news_lg")

# Load the German BERT tokenizer and model
model_name_de = "bert-base-german-cased"
model_de = AutoModel.from_pretrained(model_name_de)
tokenizer_de = AutoTokenizer.from_pretrained(model_name_de)


#Total tokens without stop words
def get_text_tokens(text, stopwords=[]):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if ((token.text and token.lemma_ and token.lemma_ != ' '
                                                      and token.text not in stopwords
                                                      and token.pos_ not in ['PUNCT', 'NUM']))]

#MATTR (moving window=50)
def calculate_mattr(tokens: List[str], window_size: int = 50) -> float:
    if len(tokens) < window_size:
        return len(set(tokens)) / len(tokens)  # Use full length if shorter than the window

    ttr_values = []
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i + window_size]
        ttr = len(set(window)) / window_size
        ttr_values.append(ttr)
    
    return np.mean(ttr_values)

#Yngve score of syntactic complexity
def calculate_yngve_score(doc):
    depth_text = []
    for sent in doc.sents:
        max_depth = 0
        for token in sent:
            depth = 0
            # Traverse the ancestors of the token to find depth
            for ancestor in token.ancestors:
                depth += 1
            if depth > max_depth:
                max_depth = depth
        depth_text.append(max_depth)
    return np.mean(depth_text)

#LCC=measure of graph connectivity
class NaiveGraph:
    @staticmethod
    def _text2graph(words: List[str]) -> nx.MultiDiGraph:
        gr = nx.MultiDiGraph()
        gr.add_edges_from(zip(words[:-1], words[1:]))
        return gr

    @staticmethod
    def get_LCC(words: List[str]) -> int:
        graph = NaiveGraph._text2graph(words)
        lcc = len(max(nx.weakly_connected_components(graph), key=len))
        return lcc

def moving_LCC(words: List[str], window_size: int = 100) -> float:
    if len(words) <= window_size:
        return NaiveGraph.get_LCC(words)
    else:
        lcc_values = []
        for i in range(len(words) - window_size):
            window = words[i:i + window_size]
            lcc_values.append(NaiveGraph.get_LCC(window))
        return np.mean(lcc_values)

#TF-IDF weighted sentence vectors
def vectorize_sent(text, oov=oov, stopwords=fillers):
    doc = nlp(text)
    tokens = [token for token in doc if ((token.text and token.lemma_ and token.lemma_ != ' '
                                          and token.text not in stopwords
                                          and not token.is_oov
                                          and token.pos_ not in ['PUNCT', 'NUM']))]
    return [token.text for token in tokens], [token.vector for token in tokens]

def idf_sent_vectors(words: List[str], vectors: List[np.array], lang='de') -> np.array:
    assert len(words) > 0
    assert len(words) == len(vectors)
    weights = [word_frequency(w, lang=lang) for w in words]
    if sum(weights) == 0:  
        return np.average(vectors, axis=0)
    return np.average(vectors, axis=0, weights=weights)

def vectorize_sents(sents, stopwords=fillers, use_tf_idf=True, lang='de'):
    sentence_vectors = []
    for s in sents:
        sent_tokens, sent_vectors = vectorize_sent(s, stopwords=stopwords)
        if not sent_tokens:
            continue
        if use_tf_idf:
          sent_vector = idf_sent_vectors(sent_tokens, sent_vectors, lang=lang)
        else:
          sent_vector = np.mean(sent_vectors, axis=0)
        sentence_vectors.append(sent_vector)
    return sentence_vectors

#local coherence
def cos_sim(v1, v2):
    return np.inner(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

def get_local_coherence(clause_vectors):
    if len(clause_vectors) <= 1:
        return [np.nan]
    local_coherence_list = []
    for i in range(len(clause_vectors)-1):
        local_coherence_list.append(cos_sim(clause_vectors[i], clause_vectors[i+1]))
    return local_coherence_list

#BERT next sentence probability
tokenizer_nsp_de = BertTokenizer.from_pretrained(model_name_de)
model_nsp_de = BertForNextSentencePrediction.from_pretrained(model_name_de)

def next_sent_prob(sent_text_1: str, sent_text_2: str,
                   tokenizer_nsp=tokenizer_nsp_de,
                   model_nsp=model_nsp_de) -> float:
  tokenized = tokenizer_nsp(sent_text_1, sent_text_2, return_tensors='pt')
  predict = model_nsp(**tokenized)
  pred = torch.nn.functional.softmax(predict.logits[0], dim=0)[0].item()
  return pred


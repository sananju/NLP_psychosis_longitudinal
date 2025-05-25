#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Code to preprocess speech transcripts for subsequent NLP analysis, as in Just et al., 2025:
DOI to follow.
Please cite the paper above if you use this code for your own work.
Authors Galina Ryazanskaya and Sandra Just 25/05/2025
"""

import re
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

#Define list of stopwords and fillers in German
stopwords = stopwords.words('german')
fillers = ['Ja', 'ja', 'Joa', 'joa', 'Ähm', 'ähm', 'Äh', 'äh', 'Oh', 'oh', 'Ohje', 'ohje',  'Hm', 'hm', 'Mh', 'mh', 'Mhm', 'mhm', 'Na', 'na', 'Naja', 'naja', 'Ne', 'ne', 'EM', 'em', 'Ehm', 'ehm', '[unv]', '(...)', '(..)', '(.)', '[...]', '[..]', '[.]']

#Remove timestamps
def remove_timestamps(text):
    return re.sub('#\d\d:\d\d:\d\d-\d#', '', text)

#Remove hanging punctuation
def remove_hanging_punct(text):
    text = re.sub('\?+', '?', text)
    text = text.replace('. .', '.').replace('. ,', '.').replace(': .', ':')
    if text.startswith('.'):
        text = text.strip('. ')
    text = re.sub('\s+([\.\,\:\?\(\)!\]\[])', r"\g<1>", text)
    text = re.sub(':\s+, ', ': ', text)
    return text

#Remove punctuation that is not needed in analysis
def remove_unwanted_punct(text):
    text = re.sub("""[\(\[]\.+[\)\]]|\[ ?unv\.?]|['`«»е]|\(? ?…\)?""", '', text)
    return text

#Clean stopwords from transcript
def clean_stopwords(text, stopwords=[]):
    tokens = nltk.word_tokenize(text) 
    tokens = [w for w in tokens if not w in stopwords] 
    text_cleaned = ' '.join(tokens)
    return text_cleaned

#Clean verbal fillers from transcript
def clean_fillers(text, fillers=[]):
    tokens = nltk.word_tokenize(text) 
    tokens = [w for w in tokens if not w in fillers] 
    text_cleaned = ' '.join(tokens)
    return text_cleaned

#Combine to one preprocessing function
def preprocess(text, fillers=fillers, stopwords=stopwords): 
    text = remove_timestamps(text)
    text = remove_hanging_punct(text)
    text = remove_unwanted_punct(text)
    text = clean_stopwords(text, stopwords=stopwords)
    text = clean_fillers(text, fillers=fillers)
    return text


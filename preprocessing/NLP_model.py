import nltk
from collections import Counter
import math
import joblib
import json

file_path = "C:/Users/cy028986/apu/NLP/assignment/streamlit/corpus_cleaned_tokens.json"

with open(file_path, 'r') as file:
    corpus = json.load(file)

tokenized_text = [list(map(str.lower, corpus))]

# Preprocess the tokenized text for 3-grams language modelling
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.smoothing import KneserNey
from nltk.lm import KneserNeyInterpolated

n = 2 #2 for bigram
train_data, vocab_data = padded_everygram_pipeline(n, tokenized_text)

lm = KneserNeyInterpolated(n)

print('model building start')
lm.fit(train_data, vocab_data)

joblib.dump(lm, 'C:/Users/cy028986/apu/NLP/assignment/streamlit/lm_model.pkl') 
print('complete')


#***********************************************************************************************************************#

#***********************************************************************************************************************#

#from nltk.corpus import words, brown
#from nltk.util import ngrams

#from nltk.util import bigrams
#from nltk.util import everygrams
#from nltk.util import pad_sequence
#from nltk.lm.preprocessing import pad_both_ends
#from nltk.lm import MLE # Maximum Likelihood Estimation - unsmoothed LM

#nltk.download('brown')  #don't repeat if already download
#brown_lc_words = [wd.lower() for wd in brown.words()]
#WORDS = brown_lc_words

# Dictionary from NLTK
#nltk.download('words')  #don't repeat if already download

#dictionary = set(words.words())

#word_freq = Counter(brown_lc_words) #normalized_tokens)
#total_words = sum(word_freq.values())

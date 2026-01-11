import nltk
from collections import Counter
import math
import joblib
import json
import re

from nltk.lm import KneserNeyInterpolated
from nltk.lm.vocabulary import Vocabulary
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
from nltk.util import bigrams

#***********************************************************************************************
#Tokenizer function
#***********************************************************************************************

def tokenizer (text):
    text = re.sub(r"[“”]", '"', data)
    text = re.sub(r"[’]", "'", data)

    # Tokenize with re.split()
    pattern = r"([@#]?\w+|:[\w_]+:|[!?.,…]+|[=]+|[^\w\s])"
    raw_tokens = re.split(pattern, text)
    tokens = [tok.strip() for tok in raw_tokens if tok.strip()]

    # Remove punctuation and apostrophes
    clean_tokens = []
    for t in tokens:
        cleaned = re.sub(r"[^a-zA-Z@']+", "", t) #clean token only has text
        if cleaned:
            clean_tokens.append(cleaned)

    # Normalize to lowercase
    normalized_tokens = [t.lower() for t in clean_tokens]

    # Join tokens temporarily to merge split contractions (e.g. auditors ' → auditors')
    joined_text = " ".join(normalized_tokens)
    joined_text = re.sub(r"\b(\w+)\s*'\s*(\w+)\b", r"\1' \2", joined_text)

    # Correct all possible apostrophe errors  
    replace_text = joined_text.replace("s' s", "s'_s")
    replace_text = replace_text.replace("' s", "'s")
    replace_text = replace_text.replace("s'_s", "s' s")
    replace_text = replace_text.replace(" s ", " ")

    # Remove all roman numeric letters
    pattern = r'\b(ii|iii|iv|vi|vii|viii|ix|xi|xii|xiii)\b'
    replace_text = re.sub(pattern, '',replace_text, flags=re.IGNORECASE)

    # Remove single letter words except "a" and "i"
    pattern = r'\b(b|c|d|e|f|g|h|j|k|l|m|n|o|p|q|r|t|u|v|w|x|y|z)\b'
    replace_text = re.sub(pattern, '',replace_text)
    return replace_text

#******************************************************************************************
#Model Perplexity calculation
#******************************************************************************************
"""
file_path = "C:/Users/cy028986/apu/NLP/assignment/streamlit/corpus_cleaned_tokens.json"

with open(file_path, 'r') as file:
    corpus = json.load(file)

train_data = [list(map(str.lower, corpus))]

vocab = Vocabulary(
    [word for document in train_data for word in document]) #,unk_cutoff=1)

n=2

train, _ = padded_everygram_pipeline(n, train_data)
lm = KneserNeyInterpolated(n)
lm.fit(train, vocab)

# Test data
file = open('C:/Users/cy028986/apu/NLP/assignment/text_perplexity.txt', encoding='utf-8')
data = file.read()
#print(test_sentences)

test_sentence = tokenizer(data).split()

padded_sentence = list(pad_both_ends(test_sentence, n=2))
test_grams = list(bigrams(padded_sentence))

ent = lm.perplexity(test_grams)
print("Perplexity: " + str(ent))


"""

#******************************************************************************************
#Model accuracy calculation
#******************************************************************************************
#file = open('C:/Users/cy028986/apu/NLP/assignment/original_text.txt', encoding='utf-8')
#data = file.read()
data = "Shareholders are reminded to ensure that their participation in the Dividend Reinvestment Plan will not result in a breach of any restrictions on their respective holding of Gamuda Shares which may be imposed by their contractual obligations, or by any statute, law or regulation in force in Malaysia or any other relevant jurisdiction, or by any relevant authorities (unless the requisite approvals under the relevant statute, law or regulation or from the relevant authorities are first obtained"
original = tokenizer(data).split()
print(len(original))
#file = open('C:/Users/cy028986/apu/NLP/assignment/suggested_text.txt', encoding='utf-8')
#data = file.read()
data = "Shareholder are reminded to ensure that their participation in the Dividend Revinvestment Plan will not result in a break of any restrictions on their respective holding of Gamuda Shares which may be imposed by their contractual obligations, or by any statue, law or regulation in force in Malaysia or any other relevant jurisdiction, or by any relevant authorities (unless the requisite approval under the relevant statue, law or regulation or from the relevant authorities are first obtained"
suggested = tokenizer(data).split()
print(len(suggested))

correct_cnt = 0
tot = len(original)
if tot == len(suggested):
    for i in range(tot):
        if suggested[i] == original[i]:
            correct_cnt += 1
        else:
            pass
    acc=correct_cnt/tot
    print("Correct number of words: " + str(correct_cnt))
    print("Total number of words: " + str(tot))
    print("Accuracy: " + str(acc))
    
else:
    print("Number of words not tally")
        
#"""

#*****************************************************************************#
#Libraries
#*****************************************************************************#
import streamlit as st
import nltk
from nltk.corpus import words, brown
from collections import Counter

import math
import requests
import json
import random
import itertools
import random
import re
import time

import joblib
from io import BytesIO

from pyxdameraulevenshtein import damerau_levenshtein_distance
from iteration_utilities import random_product

#*****************************************************************************#
#Initiate corpus 
#*****************************************************************************#

file_url = "https://raw.githubusercontent.com/chyong007/nlp-spell-checker-app/refs/heads/main/corpus_cleaned_tokens.json"

with requests.get(file_url) as file:
    corpus = json.loads(file.text)

dictionary = set(corpus) #

word_freq = Counter(corpus)
total_words = sum(word_freq.values())

#*****************************************************************************#
#Begin streamlit app design
#1.0 Input and Dictionary
#*****************************************************************************#
st.title("NLP: Spell & Grammar Checker")
st.success("Text Input and Dictionary")

# 1.1 Initialize session state
if 'content' not in st.session_state:
    st.session_state['content'] = " "
if 'pill_key' not in st.session_state:
    st.session_state['pill_key'] = []
if 'non_word' not in st.session_state:
    st.session_state['non_word'] = []
if 'real_word' not in st.session_state:
    st.session_state['real_word'] = " "
if 'total_time' not in st.session_state:
    st.session_state['total time'] = " "

#1.2 Input text area
col1, col2 = st.columns([3,1])
with col1:
    input = st.text_area("Enter your text for spelling check (numbers will removed)",
                        height=245,
                        max_chars = 500,
                        key="user_input",
                        on_change=None)

sent = input.split()
lw_input=[wd.lower() for wd in sent]

clean_tokens = []
for t in lw_input:
    cleaned = re.sub(r"[^a-zA-Z@#_']+", "", t) #a-zA-Z0-9@#:_
    if cleaned:
        clean_tokens.append(cleaned)

normalized_tokens = [t.lower() for t in clean_tokens]

joined_text = " ".join(normalized_tokens) # Join tokens temporarily to merge split contractions (e.g. auditors ' → auditors')
joined_text = re.sub(r"\b(\w+)\s*'\s*(\w+)\b", r"\1' \2", joined_text)

# Correct all possible apostrophe errors  
replace_text = joined_text.replace("s' s", "s'_s")
replace_text = replace_text.replace("' s", "'s")
replace_text = replace_text.replace("s'_s", "s' s")
replace_text = replace_text.replace(" s ", " ")

pattern = r'\b(ii|iii|iv|vi|vii|viii|ix|xi|xii|xiii)\b' # Remove all roman numeric letters
replace_text = re.sub(pattern, '',replace_text, flags=re.IGNORECASE)

pattern = r'\b(b|c|d|e|f|g|h|j|k|l|m|n|o|p|q|r|t|u|v|w|x|y|z)\b' # Remove single letter words except "a" and "i"
replace_text = re.sub(pattern, '',replace_text)

final_tokens = replace_text.split() # Split again into final clean tokens

#1.3 Create dictionary and search engine
dict=sorted(list(dictionary))

with col2:
    search = st.text_input("Search word in dictionary")
    
    matching_items = [s for s in dict if search in s]
    
    dicta = st.text_area("Words in dictionary",
                        height=155,
                        value="\n".join(matching_items),
                        )
 
#*****************************************************************************#
#2.0 Create calculation form
#*****************************************************************************#   
output=[]

with st.form(key='my_form'):
    button1=st.form_submit_button('Check !')
    if button1:
        st.session_state['content'] = ' '
                
        def P_word(word):
            """Prior probability of a word."""
            return word_freq[word.lower()] / total_words if word.lower() in word_freq else 1e-9

        def P_misspelling_given_word(misspelled, candidate):
            """Likelihood: exponential decay with distance."""
            dist = damerau_levenshtein_distance(misspelled, candidate)
            return math.exp(-dist)  # strong penalty for larger distances

        def correct_spelling_bayes(misspelled, dictionary, max_suggestions=3, max_distance=2):
            scores = []
            for w in dictionary:
                if misspelled not in dictionary:
                    tag1 = 0  #non-word error
                else:
                    tag1 = 1
                dist = damerau_levenshtein_distance(misspelled, w)
                if dist <= max_distance:  # 2 for edit distance
                    prior = P_word(w)
                    likelihood = P_misspelling_given_word(misspelled, w)
                    posterior = prior * likelihood
                    scores.append((w, prior, likelihood, posterior, tag1))
    
            scores.sort(key=lambda x: x[3], reverse=True)  # sort by posterior
            return scores[:max_suggestions]

        #2.1 Start inferencing process
        start_time = time.time()
        with st.spinner("Waiting for model inference...", show_time=True):
   
            cands = []
            cands1 = []
            non_word_suggest=[]
            non_word_suggest1=[]
            non_word_err=[]
   
        #2.2 Candidates from edit distance    
            for i in final_tokens:
                suggestions = correct_spelling_bayes(i, dictionary)
                if len(suggestions) == 0:
                    non_word_err.append(i)
                    non_word_suggest1.append((i,["**No available suggestions**"]))
                    cands1.append([i])
                    i = ":blue-badge[" + i + "]"
                    output.append(i)
                else:
                    for sublist in suggestions:
                        cands.append(sublist[0]) #sublist[0] is word candidate
                        if sublist[4] == 0: #sublist[4] is tag for non-word error
                            non_word_suggest.append(sublist[0]) 
                    non_word_suggest1.append((i,non_word_suggest))
                    if sublist[4] == 0:
                        non_word_err.append(i)
                        i = ":red-badge[" + i + "]"
                    cands1.append(cands)
                    output.append(i)
                    cands = []
                    non_word_suggest=[]
            st.session_state['pill_key'] = non_word_err
            st.session_state['non_word'] = non_word_suggest1
        
        #2.3 Possible matrix of candidates
            first_elements = [sublist[0] for sublist in cands1] #capture all first elements 
            
            if len(sent) > 5:
                random.seed(50)
                cands2 = random_product(*cands1, repeat=500)
                n = len(sent)
                batches = [cands2[i:i+n] for i in range(0, len(cands2), n)]
                list_of_lists =[first_elements]

                special = list(set(batches))
                candidates = special + list_of_lists
            else:
                batches = itertools.product(*cands1)
                candidates = list(set(batches))

            file_url = "https://raw.githubusercontent.com/chyong007/nlp-spell-checker-app/main/lm_model.pkl"
            with requests.get(file_url) as file:
                model = joblib.load(BytesIO(file.content)) 
  
            def bigram_prob(sent_arg):
                if not sent_arg:
                    return 0.0

        #2.4 Inference candidates with language model
                prob = P_word(sent_arg[0]) #Initialize with the prior probability of the first word

                for i in range(1, len(sent_arg)):
                    prev_word = sent_arg[i-1]
                    current_word = sent_arg[i]

                    score = model.score(current_word, prev_word.split())
                    conditional_prob = score
                    prob *= conditional_prob
                return prob
            
        #2.5 Probability for all word candidates
            probability=[]
            
            for i in candidates:
                probability.append((i, bigram_prob(i)))
            probability.sort(key=lambda x: x[1], reverse=True)
                          
            real_err=[]
            for word, prob in probability[:5]:
                real_err.append(' '.join(word))

            multiline_real = "\n\n".join(real_err)
            st.session_state['real_word'] = multiline_real

        end_time = time.time()
        total_time = end_time - start_time
        st.session_state['total_time'] = total_time
      
#*****************************************************************************#
#3.0 Display results
#*****************************************************************************#   
if len(sent) > 0:
    st.success("Spelling check complete!" + "\n\n" +
                f"Task completed in {st.session_state['total_time']:.2f} seconds for total words of " + str(len(sent)))     
else:
    st.success("Spelling check!")
    
content = st.session_state['content']

for word in output:
    if word ==[]:
        pass 
    else:
        content += word + " "

#3.1 Display result of text investigation for non-word error
st.session_state['content'] = content
with st.container(border=True):
    st.markdown(st.session_state['content'], unsafe_allow_html=True)

col3, col4 = st.columns([1,1])

#3.2 Display pill buttons for all non-word errors
with col3:
    selection = st.pills("Non-word error(s)", options=st.session_state['pill_key'], selection_mode="single", default=None, key="pills_key")

    if selection:
        selected=''.join({selection})
        non_word_suggest2=[]    
    
        for misspell, word in st.session_state['non_word']:
            if misspell == selected:
                non_word_suggest2=", ".join(word)

        st.text_area("Top maximum 3 non-word correction", 
                    value=non_word_suggest2,
                    height=123,
                    disabled=False)
    else:
        pass

#3.3 Display all possibility for real-word errors
with col4:
    st.text_area("Top 5 possible real word correction", 
                height=295,
                value=st.session_state['real_word'],
                disabled=False)

#*****************************************************************************#
#4.0 Program ends
#*****************************************************************************#   

st.write("Streamlit Version:", st.__version__)
















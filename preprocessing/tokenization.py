import nltk
import re
import json

file = open('C:/Users/cy028986/apu/NLP/assignment/assignment_corpus.txt', encoding='utf-8')
data = file.read()
#data = "you have and/or"
#print("Total words in data: " + str(len(data.split())))

#*********************************************************************************************

# Normalize fancy quotes
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

# Split again into final clean tokens
final_tokens = replace_text.split()

# Display results
print(final_tokens)
print("Total words in corpus: " + str(len(final_tokens)))

file_path = 'corpus_cleaned_tokens.json'

with open(file_path, 'w', encoding='utf-8') as f:
#    # Use 'indent=4' for human-readable, pretty-printed output
    json.dump(final_tokens, f, indent=4)

print("json done")
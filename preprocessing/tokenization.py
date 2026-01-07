import nltk
import re
import json


file = open('C:/Users/cy028986/apu/NLP/assignment/assignment_corpus.txt', encoding='utf-8')
data = file.read()
data

#print("Total words in data: " + str(len(data.split())))

#*********************************************************************************************

# Normalize fancy quotes
text = re.sub(r"[“”]", '"', data)
text = re.sub(r"[’]", "'", data)

# Tokenize with re.split()
pattern = r"([@#]?\w+|:[\w_]+:|[!?.,…]+|[=]+|[^\w\s])"
raw_tokens = re.split(pattern, text)
tokens = [tok.strip() for tok in raw_tokens if tok.strip()]

# Remove punctuation but keep apostrophes
clean_tokens = []
for t in tokens:
    cleaned = re.sub(r"[^a-zA-Z@']+", "", t) #clean token only has text
    if cleaned:
        clean_tokens.append(cleaned)

# Join tokens temporarily to merge split contractions (e.g. don ' t → don't)
joined_text = " ".join(clean_tokens)
joined_text = re.sub(r"\b(\w+)\s*'\s*(\w+)\b", r"\1'\2", joined_text)

# Split again into final clean tokens
final_tokens = joined_text.split()

# Normalize to lowercase
normalized_tokens = [t.lower() for t in final_tokens]

# Display results
#print("Total words in corpus: " + str(len(normalized_tokens)))

file_path = 'corpus_cleaned_tokens.json'

with open(file_path, 'w', encoding='utf-8') as f:
    # Use 'indent=4' for human-readable, pretty-printed output
    json.dump(normalized_tokens, f, indent=4)

print("json done")
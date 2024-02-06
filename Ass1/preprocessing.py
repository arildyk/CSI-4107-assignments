import os
import re
import nltk
from nltk.stem import PorterStemmer

#! do pip install nltk on your terminal if you don't have nltk. 
#! Don't forget to also uncomment the comment below before running (just do it once).
# nltk.download('punkt')

def read_doc_and_tokenize(doc, tag_name):
    # Extract content within the specified tag
    #* Extraire le contenu dans le tag spécifiée
    pattern = f'<{tag_name}>(.*?)</{tag_name}>'
    matches = re.findall(pattern, doc, re.DOTALL | re.IGNORECASE)

    if matches:
        for i in range(len(matches)):
            text = matches[i]

            # Remove HTML tags.
            # *Supprimez les tags HTML.
            html_free_content = re.sub(r"<[^>]+>", " ", text)

            # Remove punctuation and numbers, replace with space.
            #* Supprimez la ponctuation et les chiffres, remplacez-les par un espace.
            polished_text = re.sub(r"[^a-zA-Z\s]", " ", html_free_content)

            # Tokenize and convert to lowercase.
            #* Tokeniser et convertir en lettres minuscules.
            tokens = nltk.word_tokenize(polished_text)
            tokens_lower = [token.lower() for token in tokens]

        return tokens_lower
    else:
        return []

def stem(tokens):
    # Stem each token in tokens.
    #* Stem chaque token dans le tableau tokens.
    p = PorterStemmer()
    stemmed_tokens = [p.stem(token) for token in tokens]
    return stemmed_tokens

def remove_stop_words(tokens):
    # Remove each stop word in tokens.
    #* Supprimez chaque stop word dans le tableau tokens.
    with open("stop_words.txt", 'r', encoding='utf-8') as file:
        content = file.read()
        stop_words = nltk.word_tokenize(content)
        tokens = [token for token in tokens if token not in stop_words]
    return tokens


# def convert_files(input_folder, output_folder):
#     for filename in os.listdir(input_folder):
#         input_path = os.path.join(input_folder, filename)

#         if os.path.isfile(input_path):
#             with open(input_path, 'r', encoding='utf-8') as input_file:
#                 file_content = input_file.read()

#                 output_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}.txt')
#                 with open(output_path, 'w', encoding='utf-8') as output_file:
#                     output_file.write(file_content)

vocabulary = []

for filename in os.listdir(os.getcwd() + "/coll1/"):
    file_path = os.path.join(os.getcwd() + "/coll1/", filename)

    with open(file_path, 'r', encoding='utf-8') as file:
        document = file.read()

        # Preprocess words within the TEXT tags.
        #* Prétraitez les mots dans les tags TEXT.
        words_text = remove_stop_words(stem(read_doc_and_tokenize(document, 'TEXT')))

        # Preprocess words within the HEAD tags.
        #* Prétraitez les mots dans les tags HEAD.
        words_head = remove_stop_words(stem(read_doc_and_tokenize(document, 'HEAD')))

        # Remove potential duplicate words by combining words_head and words_text into a set.
        #* Supprimez les mots en double potentiels en combinant words_head et words_text dans un ensemble.
        words = set(words_head + words_text)

        # Sort the set by converting it back into a list.
        #* Triez l'ensemble en le reconvertissant en liste.
        words_sorted = sorted(list(words))

        # Add the sorted list into vocabulary.
        #* Ajoutez la liste triée dans le vocabulaire.
        vocabulary += words_sorted

# Remove potential duplicate words in the vocabulary and turn it to a list to sort it.
#* Supprimez les mots en double potentiels dans le vocabulaire et transformez-le en liste pour le trier.        
vocabulary_set = set(vocabulary)
vocabulary = sorted(list(vocabulary_set))

with open("vocabulary.txt", 'w', encoding='utf-8') as output_file:
    for word in vocabulary:
        output_file.write(word + '\n')

# convert_files(os.getcwd() + "/coll/", os.getcwd() + "/coll1/")

print(vocabulary)
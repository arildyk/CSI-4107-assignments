import os
import re
import nltk
from nltk.stem import PorterStemmer
import time

#! do pip install nltk on your terminal if you don't have nltk. 
#! Don't forget to also uncomment the comment below before running (just do it once).
# nltk.download('punkt')

def clean_text(text):
    # Normalize hyphenated words
    text = re.sub(r'-\s*', '', text)
    # Normalize quotation marks
    text = re.sub(r'[\u2018\u2019\u201C\u201D]', '"', text)
    # Remove punctuation and numbers (keeping underscores in this example)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return text

def read_file_and_tokenize(file, tag_name):
    # Extract content within the specified tag
    pattern = f'<{tag_name}>(.*?)</{tag_name}>'
    matches = re.findall(pattern, file, re.DOTALL)

    tokens_final = []
    
    if tag_name == "DOCNO":
        return matches
    
    for text in matches:
        # Remove HTML tags.
        html_free_content = re.sub(r"<[^>]+>", " ", text)

        # Remove punctuation and numbers, replace with space.
        polished_text = clean_text(html_free_content)

        # Tokenize and convert to lowercase.
        tokens = nltk.word_tokenize(polished_text)

        for token in tokens:
            tokens_final.append(token)

    return tokens_final

def stem(tokens):
    # Stem each token in tokens.
    p = PorterStemmer()
    stemmed_tokens = [p.stem(token) for token in tokens]
    return stemmed_tokens

def load_stop_words():
    # Load the stop words.
    with open("stop_words.txt", 'r') as file:
        content = file.read()
        return set(nltk.word_tokenize(content))

def remove_stop_words(stop_words, tokens):
    return [token for token in tokens if token not in stop_words]


if __name__ == "__main__":

    start_time = time.time()

    vocabulary = []

    stop_words = load_stop_words()

    for filename in os.listdir(os.getcwd() + "/coll1/"):
        file_path = os.path.join(os.getcwd() + "/coll1/", filename)

        print("doing " + filename + "...")

        with open(file_path, 'r') as file:
            document = file.read()

            # Preprocess words within the TEXT tags.
            words_text = remove_stop_words(stop_words, stem(read_file_and_tokenize(document, 'TEXT')))

            # Preprocess words within the HEAD tags.
            words_head = remove_stop_words(stop_words, stem(read_file_and_tokenize(document, 'HEAD')))

            # Remove potential duplicate words by combining words_head and words_text into a set.
            words = set(words_head + words_text)

            # Sort the set by converting it back into a list.
            words_sorted = sorted(list(words))

            # Add the sorted list into vocabulary.
            vocabulary += words_sorted

    # Remove potential duplicate words in the vocabulary and turn it to a list to sort it.       
    vocabulary_set = set(vocabulary)
    vocabulary = sorted(list(vocabulary_set))

    with open("vocabulary.txt", 'w') as output_file:
        for word in vocabulary:
            output_file.write(word + '\n')

    end_time = time.time()
    ex_time = end_time - start_time

    print(f"Execution time: {ex_time} seconds")
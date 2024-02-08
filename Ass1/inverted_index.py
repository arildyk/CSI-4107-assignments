import json
import os
import re

from preprocessing import read_file_and_tokenize, remove_stop_words, stem

def build_inverted_index():

    vocabulary = []

    with open("vocabulary.txt", 'r') as file:
        for token in file:
            vocabulary.append(token.strip())

    index = {word: {} for word in vocabulary}
    
    for filename in os.listdir(os.getcwd() + "/coll1/"):
        file_path = os.path.join(os.getcwd() + "/coll1/", filename)

        with open(file_path, 'r', encoding='utf-8') as file:
            textfile = file.read()

            documents = re.findall(r'<DOC>(.*?)<\/DOC>', textfile, re.DOTALL)

            for d in documents:
                doc_no = read_file_and_tokenize(d, 'DOCNO')[0].strip()

                tokens_text = remove_stop_words(stem(read_file_and_tokenize(d, 'TEXT')))
                tokens_head = remove_stop_words(stem(read_file_and_tokenize(d, 'HEAD')))

                tokens = list(tokens_head + tokens_text)

                for token in tokens:
                    for word in vocabulary:
                        if word in token:
                            if doc_no in index[word]:
                                index[word][doc_no] += 1
                            else:
                                index[word][doc_no] = 1
    
    with open('index.json', 'w') as json_file:
        json.dump(index, json_file, indent=4)

    print(index)

build_inverted_index()
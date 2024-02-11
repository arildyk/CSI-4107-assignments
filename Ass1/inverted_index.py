import json
import os
import re
import time

from preprocessing import load_stop_words, read_file_and_tokenize, remove_stop_words, stem

def build_inverted_index():
    number_of_docs = 0
    print("Opening vocabulary...")

    with open("vocabulary.txt", 'r') as file:
        vocabulary = set(file.read().splitlines())

    stop_words = load_stop_words()
    
    index = {word: {} for word in vocabulary}
    
    for filename in os.listdir(os.getcwd() + "/coll1/"):
        file_path = os.path.join(os.getcwd() + "/coll1/", filename)

        print("doing " + filename + "...")

        with open(file_path, 'r') as file:
            textfile = file.read()
            documents = re.findall(r'<DOC>(.*?)<\/DOC>', textfile, re.DOTALL)

            for d in documents:
                doc_no = read_file_and_tokenize(d, 'DOCNO')[0].strip()

                number_of_docs += 1

                tokens_text = remove_stop_words(stop_words, stem(read_file_and_tokenize(d, 'TEXT')))
                tokens_head = remove_stop_words(stop_words, stem(read_file_and_tokenize(d, 'HEAD')))

                tokens = tokens_head + tokens_text

                for token in tokens:
                    if token in vocabulary:
                        if doc_no in index[token]:
                            index[token][doc_no] += 1
                        else:
                            index[token][doc_no] = 1
    
    with open('index.json', 'w') as json_file:
        print("Dumping to JSON...")
        json.dump(index, json_file, indent=4)
    print("Number of documents :", number_of_docs)

if __name__ == "__main__":

    start_time = time.time()

    build_inverted_index()

    end_time = time.time()
    ex_time = end_time - start_time
    print(f"Execution time: {ex_time} seconds")

import itertools
import json
import math
import nltk
from preprocessing import load_stop_words, read_file_and_tokenize, remove_stop_words, stem

def get_idf_values(N, inverted_index):
    idf_values = {}
    for token in inverted_index:
        idf_values[token] = math.log2(N / len(inverted_index[token]))
    return idf_values

def create_doc_vectors(inverted_index):
    vecs = {}

    for token, docs in inverted_index.items():
        for doc_no, tf in docs.items():
            if doc_no in vecs:
                vecs[doc_no][token] = tf
            else:
                vecs[doc_no] = {}
    
    vecs_normalized = {}
    for doc_no, tokens in vecs.items():
        vecs_normalized[doc_no] = {token: tf / max(tokens.values()) for token, tf in tokens.items()}
    
    return vecs_normalized

def calculate_docs_tf_idf_values(document_vectors, idf_values):
    tf_idf_docs = {}

    for doc_no, tokens in document_vectors.items():
        tf_idf_docs[doc_no] = {}
        for token, tf in tokens.items():
            idf = idf_values[token]
            tf_idf = tf * idf
            tf_idf_docs[doc_no][token] = tf_idf
    
    return tf_idf_docs

def load_queries(file_path, titles_only=True):
    queries = {}
    with open(file_path, 'r') as file:
        data = file.read()
        topics = data.split('<top>')
        for topic in topics:
            if topic.strip() != '':
                num = topic.split('<num>')[1].split('<title>')[0].strip()
                title = topic.split('<title>')[1].split('<desc>')[0].strip()
                if titles_only:
                    queries[num] = {'title': title}
                else:
                    desc = topic.split('<desc>')[1].split('<narr>')[0].strip()
                    queries[num] = {'title': title, 'desc': desc}
                
    return queries

def calculate_queries_tf_idf_values(queries, idf_values, stop_words):
    tf_idf_queries = {}
    for query_num, query_info in queries.items():
        query_text = query_info['title'] + ' ' + query_info.get('desc', '')  # Combine title and description
        query_tokens = remove_stop_words(stop_words, stem(nltk.word_tokenize(query_text)))

        query_tokens_tf = {}
        for token in query_tokens:
            if token in query_tokens_tf:
                query_tokens_tf[token] += 1
            else:
                query_tokens_tf[token] = 1
        
        query_tokens_tf_normalized = {token: tf / max(query_tokens_tf.values()) for token, tf in query_tokens_tf.items()}

        query_tokens_tf_idf = {token: tf * idf_values.get(token, 0) for token, tf in query_tokens_tf_normalized.items()}

        tf_idf_queries[query_num] = query_tokens_tf_idf
    
    return tf_idf_queries

def cos_sim(doc_vec, query_vec):
    dot_product = sum(query_vec[word] * doc_vec.get(word, 0) for word in query_vec)
    query_norm = math.sqrt(sum(value ** 2 for value in query_vec.values()))
    document_norm = math.sqrt(sum(value ** 2 for value in doc_vec.values()))
    if query_norm != 0 and document_norm != 0:
        return dot_product / (query_norm * document_norm)
    else:
        return 0

def retrieve_and_rank_queries(tf_idf_docs, tf_idf_queries):
    
    results = {}
    for query_num, query_vec in tf_idf_queries.items():
        sim_scores = {}
        for doc_no, doc_vec in tf_idf_docs.items():
            sim = cos_sim(doc_vec, query_vec)
            sim_scores[doc_no] = sim
        
        ranked_docs = sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)
        results[query_num] = ranked_docs
    return results

def save_results(results, output_file):
    with open(output_file, 'w') as file:
        for query_id, ranked_documents in results.items():
            for rank, (doc_id, score) in enumerate(ranked_documents[:1000], start=1):
                file.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} nom_execution\n")


if __name__ == "__main__":

    # Charger l'index inversé depuis le fichier JSON.
    with open('index.json', 'r') as json_file:
        print("Opening JSON index file...")
        inverted_index = json.load(json_file)
        print("Done!")

    print("Getting stop words...")
    stop_words = load_stop_words()

    print("Getting idf_values...")
    idf_values = get_idf_values(79923, inverted_index)

    print("Getting document vectors...")
    doc_vecs = create_doc_vectors(inverted_index)

    print("Getting queries...")
    queries = load_queries("queries.txt", titles_only=True)

    print("Preparing document vectors...")
    tf_idf_docs = calculate_docs_tf_idf_values(doc_vecs, idf_values)

    print("Preparing query vectors...")
    tf_idf_queries = calculate_queries_tf_idf_values(queries, idf_values, stop_words)

    print("Getting similarity scores...")
    results_titles = retrieve_and_rank_queries(tf_idf_docs, tf_idf_queries)

    save_results(results_titles, 'results_titles.txt')


    
import numpy as np
import os
import re
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# read files and get the set of words used in the corpus
def get_words(dirname, regex_filter):
    nontext = regex_filter
    word_set = set()
    filenames = os.listdir(dirname)
    for filename in filenames:
        with open("news/"+filename, "r", encoding = "utf-8") as ifile:
            contents = ifile.read()
            cleaned = nontext.sub("",contents).lower()
            word_set.update(cleaned.split())
    word_list = list(word_set)
    return word_list, filenames

def tf_idf(dirname):
    nontext = re.compile(r'[^a-zA-Z\s]+')
    word_list, file_list = get_words(dirname, nontext)
    words_inverse = dict()
    for idx, word in enumerate(word_list):
        words_inverse[word] = idx

    word_matrix = np.zeros((len(file_list), len(word_list)))

    for idx, filename in enumerate(file_list):
        with open("news/"+filename, "r", encoding = "utf-8") as ifile:
            contents = ifile.read()
            cleaned = nontext.sub("",contents).lower()
            for word in cleaned.split():
                word_matrix[idx, words_inverse[word]] += 1

    idf = np.log(word_matrix.shape[0] / np.count_nonzero(word_matrix, axis=0))
    words_in_documents = np.sum(word_matrix, axis=1)
    tf = word_matrix/words_in_documents.reshape(len(words_in_documents), 1)
    tfidf = tf*idf
    return tfidf, file_list

# Find duplicates in corpus
# Parameters:
#   dirname - directory where corpus files are found
#   x_best  - return top x_best number of matches
# Returns:
#   sim_top   - each row contains indices that represent top_x matches for file given by row index
#   file_list - mapping index -> filename
def get_duplicates(dirname, x_best=50):
    tfidf, file_list = tf_idf(dirname)
    # finds the pairwise similarity between all documents in the corpus
    sim = cosine_similarity(tfidf, tfidf)
    sim_sorted = np.argsort(sim, axis=1)

    sim_top = []
    for idx, row in enumerate(sim_sorted):
        tmp = row[::-1][:x_best+1]
        # remove self-similarity
        tmp = tmp[tmp!=idx]
        sim_top.append(tmp)

    sim_top = np.array(sim_top)
    # print(sim_top[1,:])
    # print(sim_top.shape)
    return sim_top, file_list

sim_top, file_list = get_duplicates("news", 50)

print("Top match for file {}".format(file_list[1]))
print("is: {}".format(file_list[sim_top[1,0]]))

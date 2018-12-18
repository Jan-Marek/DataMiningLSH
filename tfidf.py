import numpy as np
import os
import re
from sklearn.metrics.pairwise import cosine_similarity

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
    return tfidf




# print(tf.shape)
# print(idf.shape)

tfidf = tf_idf("news")
print(tfidf.shape)
# print(word_matrix.shape)
# print(word_matrix)


# finds the pairwise similarity between all documents in the corpus
sim = cosine_similarity(tfidf, tfidf)
print(sim.shape)


from collections import defaultdict
import re
import math
import matplotlib.pyplot as plt
import numpy as np
removejunk = re.compile('([^\s\w]|_)+')

def makebag(document, terms):
    document = removejunk.sub("", document)
    ret = defaultdict(int)
    for word in document.lower().split():
        if word in terms:
            ret[word] += 1
    return ret

def get_tf_idf(bag, idf):
    baglen = sum(bag.values())
    tf = defaultdict(int)
    for word, count in bag.items():
        tf[word] = count/baglen
    tf_idf = dict()
    for word in idf:
        tf_idf[word] = tf[word]*idf[word]
    return tf_idf

def get_similarity(query_tf_idf, doc_tf_idf):
    top = sum([query_tf_idf[k]*doc_tf_idf[k] for k in query_tf_idf])
    bot = math.sqrt(sum([query_tf_idf[k]**2 for k in query_tf_idf])) * math.sqrt(sum([doc_tf_idf[k]**2 for k in doc_tf_idf]))
    return(top/bot)
          

# hopefully "dot" is not a typo of "dog"
terms = {"cat": 5, "dog": 20, "mammals": 2, "mouse": 10, "pet": 60}

query = {"mouse": 1, "cat": 1, "pet": 1, "mammals": 1}
D1 = "Cat is a pet, dog is a pet, and mouse may be a pet too."
D2 = "Cat, dog and mouse are all mammals."
D3 = "Cat and dog get along well, but cat may eat a mouse."
bags = [makebag(D1, terms), makebag(D2, terms), makebag(D3, terms)]

# compute IDF
idf = dict()
for term in terms:
#    occurences = 0
#    for bag in bags:
#        if term in bag:
#            occurences +=1
#    idf[term] = math.log(len(bags)/occurences, 2)
    idf[term] = math.log(100/terms[term])
    
print("Reported values of idf:")
for term in idf:
    print("{}: {}".format(term, idf[term]))

# compute TF
bag_tf_idfs = [get_tf_idf(b, idf) for b in bags]
query_tf_idf = get_tf_idf(query, idf)

print("\n\nResulting similarities:")
for idx, tf_idf in enumerate(bag_tf_idfs):
    print("Query-D{}: {}".format(idx+1,get_similarity(query_tf_idf, tf_idf)))
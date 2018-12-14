# DataMiningLSH
Finding similar or near-duplicate documents using LSH
**Deadline:** 21.12.2018


## Plan:
1) Find top 50 matches for each document using TF-IDF
    1) Refactor TF-IDF algorithm
    2) Find efficient way of running this for all documents
2) Implement LSH (probably minhash)
    1) Find top 50 matches using LSH
    2) Experiment with LSH configurations to approach accuraccy of TF-IDF
    3) Compare speed/accuraccy
3) Report results

## Notes:
 - do first experiments using subset of documents
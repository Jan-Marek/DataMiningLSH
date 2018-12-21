import time, argparse, sys, json
from minhash import MinHash
from lsh import MinHashLSH
import os
from nltk.tokenize import RegexpTokenizer

def read_data(folder):
    data = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            f = open(os.path.join(folder,filename))
            raw = f.read()
            tokenizer = RegexpTokenizer(r'\w+')
            data.append(list(tokenizer.tokenize(raw)))
    return data

def _compute_jaccard(x, y):
    if len(x) == 0 or len(y) == 0:
        return 0.0
    intersection = 0
    for w in x:
        if w in y:
            intersection += 1
    return float(intersection) / float(len(x) + len(y) - intersection)

def get_candidates(result):
    candidates = []
    for r in result:
        l = list(r)
        del(l[0])
        s = "".join(l)
        candidates.append(int(s))
    return candidates

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="lsh_result.json")
    args = parser.parse_args(sys.argv[1:])

    print("reading documents")
    docs = read_data("news/")
    num_perms = [32, 64, 96, 128, 160, 192, 224, 256]
    output = {"num_perms" : num_perms,
              "lsh_times" : [],
              "lsh_results" : []}

    times = []
    for num_perm in num_perms:
        print("Use num_perm = %d" % num_perm)
        start = time.clock()
        print("Running lsh on documents")
        for i in range(len(docs)):
            name = 'm'+str(i)
            locals()['m'+str(i)] = MinHash(num_perm=num_perm)
            for d in docs[i]:
                locals()['m'+str(i)].update(d.encode('utf8'))

        lsh = MinHashLSH(threshold=0.1, num_perm=num_perm)
        for i in range(len(docs)):
            lsh.insert("m{}".format(i),locals()['m'+str(i)])

        print("LSH duration:", time.clock()-start)
        for i in range(len(docs)):
            name = 'candi'+str(i)
            name = 'result'+str(i)
            locals()['result'+str(i)] = lsh.query(locals()['m'+str(i)])
            locals()['candi'+str(i)] = get_candidates(locals()['result'+str(i)])

        results = []
        for i in range(len(docs)):
            dist = {}
            for key in locals()['candi'+str(i)]:
                jac = _compute_jaccard(docs[key], docs[1])
                dist[key] = jac
            result = sorted(dist, key=dist.get, reverse=True)[1:51]
            results.append(result)

        duration = time.clock() - start
        times.append(duration)

        output["lsh_times"].append(times)
        output["lsh_results"].append(results)

        with open(args.output, 'w') as f:
            json.dump(output, f)
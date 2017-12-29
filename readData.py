import numpy as np 
import sys
from os.path import join
import pickle
f_name = sys.argv[1]
save_name = sys.argv[2]
docs = []
labels = []
voc = set()
cat = set()

f = open(f_name, 'r');
for line in f.readlines():
    label, words = line.split("\t")
    docs.append(words.split())
    labels.append(label)
    #print(len(words.split()))
    for w in words.split():
        voc.add(w)
    cat.add(label)


f.close()    

print(len(docs))
print(len(labels))
print(len(voc))
print(len(cat))

pickle.dump((docs, labels, list(voc), list(cat)), open(save_name, 'wb'))
print("Done")
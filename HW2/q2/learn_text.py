import math
import numpy as np
import sys

# Globals - vocab and categories
global vocab
global cats
vocab = []
cats = []

def read_text(f_name):
    inputs = {}

    with open(f_name, 'r') as f:
        for line in f.readlines():
            label, words = line.split("\t")

            # add this text to data
            if label not in inputs.keys():
                inputs[label] = []
            inputs[label].append(words.split())

    return inputs


def find_prior(inputs, docs):
    return float(len(docs)) / sum([len(texts) for cc, texts in inputs.items()])

def find_cc(docs):
    global vocab
    cc = []
    text_j = reduce(lambda x,y : x + y, docs)   # create one long text

    # find P(wk | currClass) for each word
    for word in vocab:
        n_k = text_j.count(word)
        cc.append(float(n_k + 1) / (len(text_j) + len(vocab)))

    return cc

def learn_NB_text(filename):
    global cats
    global vocab
    inputs = read_text(filename)

    # derive cats and vocab
    cats = inputs.keys()
    vocab = list(set([word for k, v in inputs.items() for text in v for word in text]))

    # find class_conditionals and priors:
    class_conditionals = []
    priors = []

    # calc priors
    for cat in cats:
        priors.append(find_prior(inputs, inputs[cat]))

    # calc class_conditionals
    for cat in cats:
        class_conditionals.append(find_cc(inputs[cat]))

    return class_conditionals, priors


def classify_sample(doc, cc, priors):
    global cats
    global vocab
    positions = [vocab.index(word) for word in doc if word in vocab]
    probs = []

    for class_index in range(len(cats)):
        prob = math.log(priors[class_index])
        for i in positions:
            prob += math.log(cc[class_index][i])
        probs.append(prob)

    return cats[probs.index(max(probs))]


def classify_NB_text(filename, cc, priors):
    inputs = read_text(filename)
    classified = 0
    overall = 0

    # classify all samples
    for ans, samples in inputs.items():

        # check if this sample is the same as ans
        for doc in samples:
            if ans == classify_sample(doc, cc, priors):
                classified += 1
            overall += 1

    return float(classified) / overall



def main():
    if len(sys.argv) < 3:
        print("Usage: python est_parms.py [TEST_FILE] [TRAIN_FILE] ")
        return

    cc, priors = learn_NB_text(sys.argv[1])
    suc_rate = classify_NB_text(sys.argv[2], cc, priors)
    print("Success rate:\t%f" % suc_rate)

if __name__ == "__main__":
	main()

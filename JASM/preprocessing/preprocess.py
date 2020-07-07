import sys, string
import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
import re
import json
from tqdm import tqdm

stemmer = PorterStemmer()


# nltk.download()
# load all review texts
def load_file(file):
    reviews = []
    ratings = []
    f = open(file, 'r')
    for line in f:
        l = line.strip().split('>')
        if l[0] == '<Content':
            s = str(l[1])
            reviews.append(s)
        elif l[0] == '<Rating':
            r = l[1].split('\t')
            ratings.append(int(r[0]))
    f.close()
    return reviews, ratings


def parse_to_sentence(reviews):
    review_processed = []
    actual = []
    only_sent = []
    orig_word_dict = dict()
    counter = 0
    for r in tqdm(reviews,total=len(reviews)):
        sentences = nltk.sent_tokenize(r)
        actual.append(sentences)
        sent = []
        for s in sentences:
            # words to lower case
            s = s.lower()
            # remove punctuations and stopwords
            replace_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
            s = s.translate(replace_punctuation)
            stop_words = list(stopwords.words('english'))
            additional_stopwords = ["'s", "...", "'ve", "``", "''", "'m", '--', "'ll", "'d"]
            # additional_stopwords = []
            stop_words = set(stop_words + additional_stopwords)
            # print stop_words
            # sys.exit()
            word_tokens = word_tokenize(s)
            s = [w for w in word_tokens if not w in stop_words]
            # Porter Stemmer
            stemmed =[]
            for w in s:
                stm_word = stemmer.stem(w)
                try:
                    orig_word_dict[stm_word]
                except KeyError:
                    orig_word_dict[stm_word]=w
                stemmed.append(stm_word)
            #stemmed = [stemmer.stem(w) for w in s]
            if len(stemmed) > 0:
                sent.append(stemmed)
        review_processed.append(sent)
        only_sent.extend(sent)
    if counter % 10000 == 0:
        print(counter)
    counter += 1
    return review_processed, actual, only_sent,orig_word_dict


def parse_to_sentence(reviews):
    review_processed = []
    actual = []
    only_sent = []
    orig_word_dict = dict()
    counter = 0
    for r in reviews:
        text = r
        sentences = nltk.sent_tokenize(text)
        actual.append(sentences)
        sent = []
        for s in sentences:
            # words to lower case
            s = s.lower()
            # remove punctuations and stopwords
            replace_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
            s = s.translate(replace_punctuation)
            stop_words = list(stopwords.words('english'))
            additional_stopwords = ["'s", "...", "'ve", "``", "''", "'m", '--', "'ll", "'d"]
            # additional_stopwords = []
            stop_words = set(stop_words + additional_stopwords)
            # print stop_words
            # sys.exit()
            word_tokens = word_tokenize(s)
            s = [w for w in word_tokens if not w in stop_words]
            # Porter Stemmer
            stemmed =[]
            for w in s:
                stm_word = stemmer.stem(w)
                try:
                    orig_word_dict[stm_word]
                except KeyError:
                    orig_word_dict[stm_word]=w
                stemmed.append(stm_word)
            #stemmed = [stemmer.stem(w) for w in s]
            if len(stemmed) > 0:
                sent.append(stemmed)
        review_processed.append(sent)
        only_sent.extend(sent)
    #if counter % 100 == 0:
        #print(counter)
    counter += 1
    return review_processed, actual, only_sent,orig_word_dict
# sent = parse_to_sentence(reviews)
# print( len(sent))
# print(sent[2])
import itertools

def parsesentence(sentences):
    only_sent = []
    orig_word_dict = dict()
    for s in sentences:
        # words to lower case
        s = s.lower()
        # remove punctuations and stopwords
        replace_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        s = s.translate(replace_punctuation)
        stop_words = list(stopwords.words('english'))
        additional_stopwords = ["'s", "...", "'ve", "``", "''", "'m", '--', "'ll", "'d"]
        # additional_stopwords = []
        stop_words = set(stop_words + additional_stopwords)
        # print stop_words
        # sys.exit()
        word_tokens = word_tokenize(s)
        s = [w for w in word_tokens if not w in stop_words]
        # Porter Stemmer
        stemmed =[]
        for w in s:
            stm_word = stemmer.stem(w)
            try:
                orig_word_dict[stm_word]
            except KeyError:
                orig_word_dict[stm_word]=w
            stemmed.append(stm_word)
        if len(stemmed) > 0:
            only_sent.append(stemmed)
        else:
            only_sent.append([])

    return only_sent,orig_word_dict

def create_vocab(sent,threshold):
    words = []
    # print(len(sent))
    for s in sent:
        words += s
    # print("words")
    # print(words)
    freq = FreqDist(words)
    vocab = []
    for k, v in freq.items():
        if v > threshold:
            vocab.append(k)
    # Assign a number corresponding to each word. Makes counting easier.
    vocab_dict = dict(zip(vocab, range(len(vocab))))
    return vocab, vocab_dict


def readAspectSeedWords():
    with open("./SeedWords_resturant.json") as fd:
        # with open(projectSettings + "SeedWords.json") as fd:#JSon DAta
        aspectKeywords = []
        seedWords = json.load(fd)
        for aspect in seedWords["aspects"]:
            aspectKeywords.append(aspect["keywords"])
    return aspectKeywords

def mergeSort(alist):
        if len(alist)>1:
            mid = len(alist)//2
            lefthalf = alist[:mid]
            righthalf = alist[mid:]

            mergeSort(lefthalf)
            mergeSort(righthalf)

            i=0
            j=0
            k=0
            while i < len(lefthalf) and j < len(righthalf):
                if lefthalf[i][1] < righthalf[j][1]:
                    alist[k]=lefthalf[i]
                    i=i+1
                else:
                    alist[k]=righthalf[j]
                    j=j+1
                k=k+1

            while i < len(lefthalf):
                alist[k]=lefthalf[i]
                i=i+1
                k=k+1

            while j < len(righthalf):
                alist[k]=righthalf[j]
                j=j+1
                k=k+1

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
'''vocab, vocab_dict = create_vocab(sent)
print(vocab)
print(vocab_dict)'''

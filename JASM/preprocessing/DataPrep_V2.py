from preprocess import *
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

sid = SIA()

def Remap(value, from1, to1, from2, to2):
    return (value - from1) / (to1 - from1) * (to2 - from2) + from2

def read_reviews_from_text(file_path):
    reviews = []
    count = 0
    with open(file_path, 'r') as fp:
        for line in fp:
            reviews.append(line)
            count += 1
    print("Consumed " + str(count) + " lines")
    return reviews
folder = "SemEval_15_plus_city_search"
data_file_path_test = "./Aspect_Datasets/"+folder+"/test.txt"
data_file_path_train = "./Aspect_Datasets/"+folder+"/train.txt"

print("Reading training dataset")
reviews_train = read_reviews_from_text(data_file_path_train)
reviews_train = reviews_train[:150000]
num_revs_train = len(reviews_train)

print("Number of processed train reviews " + str(num_revs_train) + " for training")

print("Reading testing dataset")
reviews_test = read_reviews_from_text(data_file_path_test)
num_revs_test = len(reviews_test)

print("Number of processed test reviews " + str(num_revs_test) + " for testing")

polarity_orig_all = []
all_reviews = reviews_train + reviews_test
for i in range(len(all_reviews)):
    all_val = []
    pol_scores = sid.polarity_scores(all_reviews[i])
    all_val.append(pol_scores['neu'])
    all_val.append(pol_scores['pos'])
    all_val.append(pol_scores['neg'])
    all_val.append(pol_scores['compound'])
    polarity_orig_all.append(all_val)
    #Textblob way
    #polarity_orig_all.append(TextBlob(all_reviews[i]).sentiment.polarity)

print("polarity_orig_all", len(polarity_orig_all))  # ,polarity_orig_all)

print("Number of all reviews " + str(len(all_reviews)))
'''
print("Parsing training reviews")
review_sent_train, review_actual_train, only_sent_train, orig_word_dict_train = parse_to_sentence(reviews_train)

print("Number of processed reviews training " + str(len(review_sent_train)))

print("Parsing testing reviews")
review_sent_test, review_actual_test, only_sent_test, orig_word_dict_test = parse_to_sentence(reviews_test)

print("Number of processed reviews testsing " + str(len(review_sent_test)))
'''

print("Parsing all reviews")
review_sent_all, review_actual_all, only_sent_all, orig_word_dict_all = parse_to_sentence(all_reviews)

print("Number of processed reviews training " + str(len(review_sent_all)))

numsent = 0
for i in range(len(review_sent_all)):
    numsent += len(review_sent_all[i])

print("numsent now " + str(numsent))

print("Training sample")
print(review_sent_all[0])
print(review_sent_all[1])

print("Testing sample")
print(review_sent_all[num_revs_train])
print(review_sent_all[num_revs_train + 1])

print("orig_word_dict training")
print(len(orig_word_dict_all))

print("review_sent")
numFreq = 5
print("Creating Vocab with at most " + str(numFreq) + " words")
vocab, vocab_dict = create_vocab(only_sent_all, numFreq)

orig_vocab = []
print("vocab size " + str(len(vocab)))
print("vocab_dict " + str(len(vocab_dict)))

extract_sentiment = 1
if extract_sentiment ==1:
    print("Extracting Sentiments for vocab words")

    orig_word_vocab_Sentiment = dict()
    for word in tqdm(vocab, total=len(vocab)):
        origword = orig_word_dict_all[word]
        orig_vocab.append(origword)
        #polarity = TextBlob(origword).sentiment.polarity
        all_val = []
        pol_scores = sid.polarity_scores(origword)
        all_val.append(pol_scores['neu'])
        all_val.append(pol_scores['pos'])
        all_val.append(pol_scores['neg'])
        all_val.append(pol_scores['compound'])
        # polarity = Remap(polarity, -1, 1, 0, 1)
        #orig_word_vocab_Sentiment[origword] = polarity
        orig_word_vocab_Sentiment[origword] = all_val

print("Building Sentiments Vector")
new_text = []
orig_all_text = []
all_sentiment_vector = []
index = 0
nzero = 0
for revSent in tqdm(review_sent_all, total=len(review_sent_all)):
    result_stem = ""
    result_orig = ""
    all_words_in_rev = []
    all_words_in_rev_orig = []
    for j in range(len(revSent)):
        for k in range(len(revSent[j])):
            all_words_in_rev.append(revSent[j][k])
            all_words_in_rev_orig.append(orig_word_dict_all[revSent[j][k]])
            result_stem += ' '.join(revSent[j])
    result_orig += ' '.join(all_words_in_rev_orig)
    sentiment_vector = []
    indo = 0
    if extract_sentiment ==1:
        for k in range(len(vocab)):
            if vocab[k] in all_words_in_rev:
                sentiment_vector.append(orig_word_vocab_Sentiment[orig_word_dict_all[vocab[k]]])
            else:
                indo += 1
                #sentiment_vector.append(0.0)
                sentiment_vector.append([0.0,0.0,0.0,0.0])
        if polarity_orig_all[index][0] == 0.0 and polarity_orig_all[index][1] == 0.0 and polarity_orig_all[index][2] == 0.0 and polarity_orig_all[index][3] == 0.0:
            indo += 1
        sentiment_vector.append(polarity_orig_all[index])
        if indo == len(vocab) + 1:
            nzero += 1
            '''
            fold = "train"
            if index>=num_revs_train:
                fold = "test"
            print("Problem with ",fold,all_words_in_rev_orig,all_words_in_rev)
            '''
        index += 1
        all_sentiment_vector.append(sentiment_vector)

    # print("sent",type(result_orig),result_orig)
    new_text.append(result_stem)
    orig_all_text.append(result_orig)
if extract_sentiment==1:
    print("N Zero", nzero)
    print("all_sentiment_vector size ", len(all_sentiment_vector), len(all_sentiment_vector[0]))

use_wordEmbedding = 0

if use_wordEmbedding==1:
    import spacy
    nlp = spacy.load('en_core_web_md')
    all_text_word_embedding = []
    print("Building Word Embedding Vector for all text")
    for line in tqdm(orig_all_text, total=len(orig_all_text)):
        doc = nlp(line)
        all_text_word_embedding.append(doc.vector)
else:
    print("Not Using WordEmbedding")

print("Building Word Count Vector")
if extract_sentiment ==1 or use_wordEmbedding ==0:
    # list of text documents
    # create the transform
    vectorizer = CountVectorizer(decode_error='ignore', vocabulary=vocab_dict)
    # tokenize and build vocab
    vectorizer.fit(new_text)
    # summarize
    print(len(vectorizer.vocabulary_))
    # print(vectorizer.vocabulary_)
    # encode document
    vector = vectorizer.transform(new_text)
    # summarize encoded vector
    print("summarize encoded vector")
    print(vector.shape)
    out = vector.toarray()
else:
    out = all_text_word_embedding

print("The whole dataset size now " + str(len(out)))
folder = "sem_eval_15_run"
text_file_path_test = "./data_text_embedding/"+folder+"/rev_text_features_new_test.txt"
sentiment_file_path_test = "./data_text_embedding/"+folder+"/rev_sentiment_features_new_test.txt"
text_file_path_train = "./data_text_embedding/"+folder+"/rev_text_features_new_train.txt"
sentiment_file_path_train = "./data_text_embedding/"+folder+"/rev_sentiment_features_new_train.txt"

text_file_handle_train = open(text_file_path_train, 'w')
text_file_handle_train.write("review_ID\tfeatures\n")
text_file_handle_test = open(text_file_path_test, 'w')
text_file_handle_test.write("review_ID\tfeatures\n")
if extract_sentiment ==1:
    sentiment_file_handle_train = open(sentiment_file_path_train, 'w')
    sentiment_file_handle_train.write("review_ID\tfeatures\n")
    sentiment_file_handle_test = open(sentiment_file_path_test, 'w')
    sentiment_file_handle_test.write("review_ID\tfeatures\n")

    print("Size of sentiment vector is ",len(all_sentiment_vector[0]))
    print("Writing the feature files")
    print("just checking ",len(all_sentiment_vector[0]),len(out[0]),len(all_sentiment_vector[1]),len(out[1]))

num_records_training = 0
num_records_testing = 0
index = 0
for line in tqdm(out, total=len(out)):
    text_feat = ""
    sent_feat = ""
    if use_wordEmbedding == 1:
        for j in range(len(all_text_word_embedding[index])):
            #text_feat += str(Remap(float(all_text_word_embedding[index][j]), -3.7, 3.1, 0, 1)) + "\t"
            text_feat += str(all_text_word_embedding[index][j]) + "\t"
    else:
        for j in range(len(line)):
            text_feat += str(line[j]) + "\t"
    # print(text_feat)
    if extract_sentiment ==1:
        for j in range(len(all_sentiment_vector[index])):
            for k in range(len(all_sentiment_vector[index][j])):
                polarity_old = all_sentiment_vector[index][j][k]
                polarity = polarity_old
                if j <len(line):
                    polarity = polarity_old *line[j]
                #polarity_map = Remap(polarity, -1, 1, 0.01, 1)
                # print(polarity_old,polarity,polarity_map)
                sent_feat += str(polarity) + "\t"

    if index >= num_revs_train:
        num_records_testing += 1
        text_file_handle_test.write(str(index - num_revs_train) + "\t" + text_feat + "\n")
        if extract_sentiment ==1:
            sentiment_file_handle_test.write(str(index - num_revs_train) + "\t" + sent_feat + "\n")
    else:
        num_records_training += 1
        text_file_handle_train.write(str(index) + "\t" + text_feat + "\n")
        if extract_sentiment == 1:
            sentiment_file_handle_train.write(str(index) + "\t" + sent_feat + "\n")

    index += 1
text_file_handle_test.close()
text_file_handle_train.close()
if extract_sentiment ==1:
    sentiment_file_handle_test.close()
    sentiment_file_handle_train.close()
print("Num training records written " + str(num_records_training))
print("Num testing records written " + str(num_records_testing))
print("Finished Data Preprocessing")
"""Reads the text data stored as sparse matrix."""
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split


def removeFirstColumn(data):
    new_data = []
    for i in range(len(data)):
        new_data.append(data[i][1:])
    new_data = np.array(new_data)
    return new_data


def LoadData(inputfile):
    print("read")
    print("loaded")
    data = np.load(inputfile, mmap_mode='r')
    print("loaded")
    print(type(data))
    print(data.shape)
    return data


def DivideTrainTestFiles():

    print("Reading Sentiment Training")
    orig_folder = "city_search_bert_run"
    orig_path = "./data_text_embedding/"+orig_folder+"/sentiment_features_train_new.npy"
    print("orig_path ",orig_path)
    sent_train = LoadData(orig_path)
    #print(sent_train.shape)
    folder = "Asp_Sent_joint_city_search_bert"
    text_unlabled = "./data/text/unlabelled/"
    sent_unlabled = "./data/image/unlabelled/"

    '''
    print("Reading Text Train")
    text_train = LoadData('./data_text_embedding/'+orig_folder+'/text_features_train_new.npy')
    print(text_train.shape)
    print("Writing text unlabled")
    file_count = 0
    file_name = "text_unlabelled-" + str(file_count) + ".npy"
    start = 0
    for i in range(len(text_train)):
        if i % 5000 == 0 and i != 0:
            file_path = text_unlabled + file_name
            np.save(file_path, text_train[start:i])
            start = i
            file_count += 1
            file_name = "text_unlabelled-" + str(file_count) + ".npy"

    file_count += 1
    file_name = "text_unlabelled-" + str(file_count) + ".npy"
    file_path = text_unlabled + file_name
    np.save(file_path, text_train[start:len(text_train) - 1])
    '''
    #'''
    print("Writing sentiment unlabled")
    file_count = 0
    file_name = "sent_unlabelled-" + str(file_count) + ".npy"
    start = 0
    for i in range(len(sent_train)):
        if i ==150000:
            break
        if i % 5000 == 0 and i != 0:
            file_path = sent_unlabled + file_name
            np.save(file_path, sent_train[start:i])
            start = i
            file_count += 1
            file_name = "sent_unlabelled-" + str(file_count) + ".npy"
	
    file_count += 1
    file_name = "sent_unlabelled-" + str(file_count) + ".npy"
    file_path = sent_unlabled + file_name
    np.save(file_path, sent_train[start:len(sent_train) - 1])
    #'''

if __name__ == '__main__':
    DivideTrainTestFiles()

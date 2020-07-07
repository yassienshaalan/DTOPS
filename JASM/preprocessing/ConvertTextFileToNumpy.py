"""Reads the text data stored as sparse matrix."""
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import pandas as pd

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


def read_text_file_check_zeros(file_path, zeros_file_path):
    reviews = []
    count = 0
    index = 0
    zero_indices = []
    print(file_path)
    zeros_handle = open(zeros_file_path, 'w')
    with open(file_path, 'r') as fp:
        for line in fp:
            if count == 0:
                count += 1
                continue
            result = line.split('\t')
            result = result[1:len(result) - 1]
            try:
                result = [float(i) for i in result]
            except ValueError:
                print(result)
            # print(len(result))
            '''
            nzeros = 0
            for val in result:
                if val == 0.0:
                    nzeros+=1
            #b = sum(result)

            if nzeros>=len(result)-3:#b == float(0):
                zeros_handle.write(str(index)+"\n")
                zero_indices.append(index)
            else:	
                reviews.append(result)
            '''
            reviews.append(result)
            # if count ==150000:
            #    break
            if count % 1000 == 0:
                print(count)
            count += 1
            index += 1
    print("Consumed " + str(count) + " lines")
    zeros_handle.close()
    return reviews, zero_indices


def read_text_file_no_check(file_path, zero_indices,heading=1,column=1):
    reviews = []
    count = 0
    index = 0

    with open(file_path, 'r') as fp:
        for line in fp:
            if heading==1 and count == 0:
                count += 1
                continue
            result = line.split('\t')
            if column==1:
                result = result[1:len(result) - 1]
            else:
                result = result[0:len(result) - 1]
            # print(count)
            if index not in zero_indices:
                result = [float(i) for i in result]
                reviews.append(result)
            
            # if count ==150000:
            #    break
            if count % 1000 == 0:
                print(count)
            count += 1
            index += 1
    print("Consumed " + str(count) + " lines")
    return reviews


def WriteNumPyAll(file_path, new_name, zeros_file_path, check_zeros, zero_indices,normalize,heading=1,column=1):
    print("Reading " + file_path)
    if check_zeros == 0:
        data = read_text_file_no_check(file_path, zero_indices,heading,column)
    else:
        data, zero_indices = read_text_file_check_zeros(file_path, zeros_file_path)
    print("Data size ",len(data),len(data[0]))
    if normalize == 1:
        print("Normalizing Data")
        data = pd.DataFrame.from_records(data)
        data.fillna(0, inplace=True)
        data = (data - data.min()) / (data.max() - data.min())
        data.fillna(0, inplace=True)
        data = data.values
        print("Size after normalization",data.shape)
    else:
        data = np.array(data)
        print(data.shape)
    print("Writing " + new_name)
    np.save(new_name, data)
    return zero_indices


if __name__ == '__main__':
    folder = 'sem_eval_16_bert_run'
    train_text_file = './data_text_embedding/' + folder + '/city_training_output_train.txt'
    test_text_file = './data_text_embedding/' + folder + '/sem_eval16_testing_output.txt'
    train_sent_file = './data_text_embedding/' + folder + '/rev_sentiment_features_new_train.txt'
    test_sent_file = './data_text_embedding/' + folder + '/rev_sentiment_features_new_test.txt'

    sent_train_out = './data_text_embedding/' + folder + '/sentiment_features_train_new'
    sent_test_out = './data_text_embedding/' + folder + '/sentiment_features_test_new'
    text_train_out = './data_text_embedding/' + folder + '/text_features_train_new'
    text_test_out = './data_text_embedding/' + folder + '/text_features_test_new'
    zeros_file_path = './data_text_embedding/' + folder + '/sent_train_zeros.txt'

    #zero_indices = WriteNumPyAll(train_sent_file, sent_train_out, zeros_file_path, 1, [],1)
    zero_indices=[]
    #WriteNumPyAll(train_text_file, text_train_out, "", 0, zero_indices,1,0,0)

    zeros_file_path = './data_text_embedding/spam_chicago_run/sent_test_zeros.txt'
    #zero_indices = WriteNumPyAll(test_sent_file, sent_test_out, zeros_file_path, 1, [],1)
     
    WriteNumPyAll(test_text_file, text_test_out, "", 0, zero_indices,1,0,0)


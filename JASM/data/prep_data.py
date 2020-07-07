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
def Test():
  print("Reading Sentiment")
  data = LoadData('sentiment_features.npy')
  print(data.shape)
  print(data[0])
  print("dropping first column")
  data = removeFirstColumn(data)
  data = data.astype(float)
  print(data.shape)
  print(data[0])
  print("Reading Text")
  data_text = LoadData('text_features.npy')
  print(data_text.shape)
  print(data_text[0])
  print("dropping first column")
  data_text = removeFirstColumn(data_text)
  data_text = data_text.astype(float)
  print(data_text.shape)
  print(data_text[0])
  print("Reading Labels")
  labels = LoadData('chicago_labels.npy')
  print(labels.shape)
  print(labels[0])
  y = removeFirstColumn(labels)
  y = y.astype(int)
  print(y[0])
  print(type(y[0]))
  revIds = []
  for i in range(len(labels)):
      revIds.append(labels[i][0])
  revIds = np.array(revIds)
  print("Revids "+str(revIds.shape))
  print(revIds[0])
  RANDOM_SEED=35
  from sklearn.model_selection import StratifiedKFold
  skf = StratifiedKFold(n_splits=5,random_state=RANDOM_SEED,shuffle=True)
  X = data
  fold_index = 0
  labels_file = "labels.npy"
  revIds_file = "revids_labels.npy"
  revIds_file_train = "revids_train_labels.npy"
  text_labled = "./data/text/labelled/text_labelled.npy"
  text_unlabled = "./data/text/unlabelled/"
  image_labled = "./data/image/labelled/image_labelled.npy"
  image_unlabled = "./data/image/unlabelled/"

  for train_index, test_index in skf.split(X, y):
      print("TRAIN:", train_index, "TEST:", test_index)
      print("train_index "+str(len(train_index)))
      print("test_index "+str(len(test_index)))
      X_train_text, X_test_text = data_text[train_index], data_text[test_index]
      X_train_sentiment, X_test_sentiment = X[train_index], X[test_index]
      y_train, y_test,revIds_train,revIds_test = y[train_index], y[test_index],revIds[train_index],revIds[test_index]
      
      count_spam = 0
      for i in range(len(y_test)):   
          if int(y_test[i][0]) == 1:
              count_spam+=1
      print("Num spam in test is "+str(count_spam)+ " out of "+str(len(y_test)))
      
      count_spam = 0
      for i in range(len(y_train)):   
          if int(y_train[i][0]) == 1:
              count_spam+=1
      print("Num spam in train is "+str(count_spam)+ " out of "+str(len(y_train)))
      
      print("Writing lables file")
      np.save(labels_file, y_test)
      
      print("Writing revids for testing file")
      np.save(revIds_file, revIds_test)

      print("Writing revids for train file")
      np.save(revIds_file_train, revIds_train)


      print("Writing text labled")
      np.save(text_labled, X_test_text)
      
      print("Writing image labled")
      np.save(image_labled, X_test_sentiment)
      
      print("Writing text unlabled")
      file_count = 0
      file_name = "text_unlabelled-"+str(file_count)+".npy"
      start = 0
      for i in range(len(X_train_text)):
          if i%10000 ==0 and i!=0:
              file_path = text_unlabled+file_name
              np.save(file_path, X_train_text[start:i])
              start = i
              file_count+=1
              file_name = "text_unlabelled-"+str(file_count)+".npy"
              
      file_count+=1
      file_name = "text_unlabelled-"+str(file_count)+".npy"
      file_path = text_unlabled+file_name
      np.save(file_path, X_train_text[start:len(X_train_text)-1])

      print("Writing sentiment unlabled")
      file_count = 0
      file_name = "image_unlabelled-"+str(file_count)+".npy"
      start = 0
      for i in range(len(X_train_sentiment)):
          if i%10000 ==0 and i!=0:
              file_path = image_unlabelled+file_name
              np.save(file_path,  X_train_sentiment[start:i])
              start = i
              file_count+=1
              file_name = "image_unlabelled-"+str(file_count)+".npy"
              
      file_count+=1
      file_name = "image_unlabelled-"+str(file_count)+".npy"
      file_path = image_unlabled+file_name
      np.save(file_path,  X_train_sentiment[start:len(X_train_sentiment)-1])
      
      '''
      valid = StratifiedKFold(n_splits=2,random_state=RANDOM_SEED,shuffle=True)
      for train_index_new, valid_index in skf.split(X_train, y_train):
          print("TRAIN:", train_index_new, "Validation:", valid_index)
          print("train_index_new "+str(len(train_index_new)))
          print("valid_index_new "+str(len(valid_index)))
          X_train_New, X_valid = X_train[train_index_new], X_train[valid_index]
          y_train_New, y_valid,revIds_train_new,revIds_Valid = y_train[train_index_new], y_train[valid_index],revIds_train[train_index_new],revIds_train[valid_index]
          print("rev_ids_train_new "+str(len(revIds_train_new))+" rev_ids_valid "+str(len(revIds_Valid)))
          
          count_spam=0
          for i in range(len(y_valid)):   
              if int(y_valid[i][0]) == 1:
                  count_spam+=1
          print("Num spam in valid is "+str(count_spam)+ " out of "+str(len(y_valid)))
          count_spam=0
          for i in range(len(y_train_New)):   
              if int(y_train_New[i][0]) == 1:
                  count_spam+=1
          print("Num spam in train is "+str(count_spam)+ " out of "+str(len(y_train_New)))
          break
      

      print("------------------------------------------------")
      fold_index+=1
      '''
      break
  #X_train, X_test = train_test_split(data, stratify=y, test_size=0.8, random_state=RANDOM_SEED)
  #print("The training shape for our encoder is " + str(len(X_train))+' '+str(len(X_train[0])))
  #print("The testing shape for our encoder is " + str(len(X_test))+' '+str(len(X_test[0])))

if __name__ == '__main__':
    Test()

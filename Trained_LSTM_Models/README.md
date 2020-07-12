#### This contains trained LSTM models for Yelp Spam datasets 
-	YelpChicago
-	YelpNY
-	YelpZip
#### Note, trained model V_0 is always a model trained on geninue data only. 
#### Later versions e.g. V_3 or so are LSTM trained itertively by removing suspected spam instances a long the way and this is the lastest trained one on the cleanest data. 
#### Each training iteration, we save a new model given a number with respect to the iteration number. For instance, if the iterative process runs for 3 iterations, the latest model will be given the number 3. 

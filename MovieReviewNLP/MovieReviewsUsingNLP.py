import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from bs4 import BeautifulSoup
import re
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

df = pd.read_csv("C:/Users/Naqib Ahmad/Downloads/labeledTrainData.tsv", delimiter="\t", quoting=3)
print(df.head())
#print(len(df))

#downloading stopwords
nltk.download('stopwords')

#data is not cleaned so we will clean it
#performing cleaning operations on a sample
sample= df.review[0]
print(sample)

#clearing HTML tags
sample= BeautifulSoup(sample).get_text()
print(sample)

#cleaning from punctuation and numbers
sample= re.sub("[^a-zA-Z]",' ', sample)
print(sample)

#converting to lowercase 
sample= sample.lower()
print(sample)

#splitting the sample data
sample= sample.split()
print(sample)

#applying stopwords cleaning operation
sw= set(stopwords.words("english"))
sample= [w for w in sample if w not in sw]
print(sample)

#now applying the cleaning to the whole dataset
def cleanProcess(review):
    review = BeautifulSoup(review).get_text()
    review = re.sub("[^a-zA-Z]", " ", review)
    review = review.lower()
    review = review.split()
    swords = set(stopwords.words("english"))
    review = [w for w in sample if w not in swords]
    return(" ".join(review))


#Now applying the cleaning function
train_x_tum = []
for r in range(len(df["review"])):
    if(r+1)%1000 == 0:
        print("Number of reviews processed = ", r+1)
    clean_review = cleanProcess(df["review"][r])
    train_x_tum.append(clean_review)
    
#train and test data
x= train_x_tum
y= np.array(df["sentiment"])

train_x, test_x, y_train, y_test = train_test_split(x,y, test_size= 0.1)
#making a matrix called bag of words
#creating a bag of words with a max of 50000 words
vectorizer= CountVectorizer(max_features=1000)
#converting train data to feature vector matrix
train_x= vectorizer.fit_transform(train_x)

#converting train data to array
train_x= train_x.toarray()
train_y= y_train

#training the random forest model
model= RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_x, train_y)

#converting test data into feature vector matrix
test_xx= vectorizer.transform(test_x)
test_xx= test_xx.toarray()

test_predict= model.predict(test_xx)
acc= roc_auc_score(y_test, test_predict)
print("Accuracy: %", acc*100)



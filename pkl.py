# Importing necessary libraries
#import numpy as np
import pandas as pd
#from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pickle



# Reading the data
data = pd.read_csv("data/final_data.csv")
target=data['label']
#print(data.title[0])
X=data['text']
y=target
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)

vectorizer = TfidfVectorizer()

# Assigning variable to the vectorizer

train_vectors = vectorizer.fit_transform(X_train)

test_vectors = vectorizer.transform(X_test)
svc_classifier=SVC()
svc_model=svc_classifier.fit(train_vectors,y_train)
#create two pickle files one for vectorizer and other for svc_classifier
pickle.dump(vectorizer,open('vectorizer.pkl','wb'))
pickle.dump(svc_classifier,open('svc_classifier.pkl','wb'))


""" #iris.drop("Id", axis=1, inplace = True)
y = iris['Classification']
iris.drop(columns='Classification',inplace=True)
X = iris[['SL', 'SW', 'PL', 'PW']]

# Training the model
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.3)
model = LogisticRegression()
model.fit(x_train,y_train)

pickle.dump(model,open('model.pkl','wb'))
 """
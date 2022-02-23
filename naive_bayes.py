import easyocr
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data= pd.read_csv('data/dataset.csv')
# print(dataset.shape)
# print(dataset.head())

language_count=data['Language'].value_counts()
print(language_count)

x = np.array(data['Text'])
y = np.array(data['Language'])
# print(x)
# print(y)

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

# print(X.shape)
# print(X_train.shape)

# Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
# nb_model.score(X_test, y_test)

y_pred = nb_model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# image_file  = 'data/Nhat2.png'

# reader = easyocr.Reader(['en','ko'])
# reader = easyocr.Reader(['en','vi'])
# reader = easyocr.Reader(['en','ja'])
# reader = easyocr.Reader(['en','th'])
# reader = easyocr.Reader(['en','ch_sim'])

# results = reader.readtext(image_file)

# text = ''
# for result in results:
#   text += result[1] + ''
# print(text)

# dataset = cv.transform([text]).toarray()
# output = nb_model.predict(dataset)
# print(output[0])

user = input("Enter a Text: ")
dataset = cv.transform([user]).toarray()
output = nb_model.predict(dataset)
print(output[0])


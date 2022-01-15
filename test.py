import easyocr
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

dataset= pd.read_csv('data/dataset.csv')
print(dataset.shape)
print(dataset.head())

language_count=dataset['Language'].value_counts()
print(language_count)

x = np.array(dataset['Text'])
y = np.array(dataset['Language'])

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(score)

image_path  = 'data/Trung1.png'

reader = easyocr.Reader(['en','vi'])
reader = easyocr.Reader(['en','ja'])
reader = easyocr.Reader(['en','ch_tra'])
reader = easyocr.Reader(['en','ch_sim'])

results = reader.readtext(image_path)

text = ''
for result in results:
  text += result[1] + ''

dataset = cv.transform([text]).toarray()
output = model.predict(dataset)
print(output[0])

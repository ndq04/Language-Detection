import codecs
# Model saving
import pickle
import re
import string

import easyocr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import feature_extraction, linear_model, pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# eng_df=pd.read_csv('data/english.csv', header=None, names=['English'])
# chi_df=pd.read_csv('data/chinese.csv', header=None, names=['Chinese'])
# kor_df=pd.read_csv('data/korean.csv', header=None, names=['Korean'])
# jp_df=pd.read_csv('data/japanese.csv', header=None, names=['Japanese'])
# vie_df=pd.read_csv('data/vietnamese.csv', header=None, names=['Vietnamese'])
# thai_df=pd.read_csv('data/thailand.csv', header=None, names=['Thailand'])

# for char in string.punctuation:
#   print(char, end=" ")
# translate_table=dict((ord(char), None) for char in string.punctuation)

# data_eng,lang_eng=[],[]
# data_chi,lang_chi=[],[]
# data_kor,lang_kor=[],[]
# data_jp,lang_jp=[],[]
# data_vie,lang_vie=[],[]
# data_thai,lang_thai=[],[]

# for i, line in eng_df.iterrows():
#   line=line['English']
#   if len(line) !=0:
#     line=line.lower()
#     line=re.sub(r'\d+','', line)
#     line=line.translate(translate_table)
#     data_eng.append(line)
#     lang_eng.append('English')

# for i, line in chi_df.iterrows():
#   line=line['Chinese']
#   if len(line) !=0:
#     line=line.lower()
#     line=re.sub(r'\d+','', line)
#     line=re.sub(r'[a-zA-Z]+','', line)
#     line=line.translate(translate_table)
#     data_chi.append(line)
#     lang_chi.append('Chinese')

# for i, line in kor_df.iterrows():
#   line=line['Korean']
#   if len(line) !=0:
#     line=line.lower()
#     line=re.sub(r'\d+','', line)
#     line=re.sub(r'[a-zA-Z]+','', line)
#     line=line.translate(translate_table)
#     data_kor.append(line)
#     lang_kor.append('Korean')

# for i, line in jp_df.iterrows():
#   line=line['Japanese']
#   if len(line) !=0:
#     line=line.lower()
#     line=re.sub(r'\d+','', line)
#     line=re.sub(r'[a-zA-Z]+','', line)
#     line=line.translate(translate_table)
#     data_jp.append(line)
#     lang_jp.append('Japanese')

# for i, line in vie_df.iterrows():
#   line=line['Vietnamese']
#   if len(line) !=0:
#     line=line.lower()
#     line=re.sub(r'\d+','', line)
#     line=line.translate(translate_table)
#     data_vie.append(line)
#     lang_vie.append('Vietnamese')

# for i, line in thai_df.iterrows():
#   line=line['Thailand']
#   if len(line) !=0:
#     line=line.lower()
#     line=re.sub(r'\d+','', line)
#     line=line.translate(translate_table)
#     data_vie.append(line)
#     lang_vie.append('Thailand')

# df=pd.DataFrame({
#     'Text':data_eng+data_chi+data_kor+data_jp+data_vie+data_thai,
#     'Language':lang_eng+lang_chi+lang_kor+lang_jp+lang_vie+data_thai
#   })
# # print(df)
# # print(df.shape)

# X,y=df.iloc[:,0], df.iloc[:,1]
# X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=0)

# # Mã hóa text
# vectorizer=feature_extraction.text.TfidfVectorizer(ngram_range=(1,3), analyzer='char')

# pipe_lr=pipeline.Pipeline([
#   ('vectorizer', vectorizer),
#   ('clf', linear_model.LogisticRegression(solver='lbfgs', max_iter=1000))
# ])
# pipe_lr.fit(X_train, y_train)

# y_pred=pipe_lr.predict(X_test)

# print(classification_report(y_test, y_pred))
# print('Accuracy: ',accuracy_score(y_test, y_pred))


# lrFile=open('LRModel.pckl','wb')
# pickle.dump(pipe_lr, lrFile)
# lrFile.close()

def lang_detect(text):
  import pickle
  import re
  import string

  import numpy as np
  translate_table=dict((ord(char), None) for char in string.punctuation)

  # Model loading
  global lrLangDetectModel
  lrLangDetectFile=open('LRModel.pckl','rb')
  lrLangDetectModel=pickle.load(lrLangDetectFile)
  lrLangDetectFile.close()

  text=' '.join(text.split())
  text=text.lower()
  text=re.sub(r'\d+','',text)
  text=text.translate(translate_table)
  pred=lrLangDetectModel.predict([text])

  return pred[0]

# image_file  = 'data/Trung.png'

# reader = easyocr.Reader(['en','ko'])
# reader = easyocr.Reader(['en','ch_sim'])
# reader = easyocr.Reader(['en','ja'])
# reader = easyocr.Reader(['en','th'])
# reader = easyocr.Reader(['en','ch_sim'])

# results = reader.readtext(image_file)

# text = ''
# for result in results:
#   text += result[1] + ''
# print('Text from image: ',text)

# output= lang_detect(text)
# print('Language: ',output)

user = input("Enter a Text: ")
output= lang_detect(user)
print('Language: ',output)

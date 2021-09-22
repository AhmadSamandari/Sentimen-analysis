import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from nltk import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

stopwords = set([x.rstrip() for x in open('stopwords.txt')])

positive_soup = BeautifulSoup(
    open('positive.review').read(),  features="lxml")
positive_soup = positive_soup.findAll('review_text')
negative_soup = BeautifulSoup(
    open('negative.review').read(),  features="lxml")
negative_soup = negative_soup.findAll('review_text')

positive_soup = positive_soup[:len(negative_soup)]
np.random.shuffle(positive_soup)

# print(positive_soup)

word_map_index1 = {}
current = 0


def tokeniz(stri):
  
    token = stri.lower()
    token = nltk.tokenize.word_tokenize(stri)
    token = [x for x in token if x not in stopwords]   #we are making list of this line words
    token = [lemmatizer.lemmatize(x) for x in token]
    token = [x for x in token if len(x) > 3]
    return token



positive_rev = []
negative_rev = []
#+/- rev are list containing lists for each row


for x in negative_soup:
    negative_token = tokeniz(x.text)
    negative_rev.append(negative_token)
    for x in negative_token:
        if x not in word_map_index1:
            word_map_index1[x] = current
            current += 1

for x in positive_soup:
    positive_token = tokeniz(x.text)
    positive_rev.append(positive_token)
    for x in positive_token:
        if x not in word_map_index1:
            word_map_index1[x] = current
            current += 1

#print(word_map_index1)

## Now lets build the vector
'''make it right
?????????????????????????????'''


def dict_to_array(item, label):
    array = np.zeros((len(word_map_index1)+1))
    for it in item:
        i = word_map_index1[it]
        array[i] += 1
        array[-1] = label
    return array


xlen = len(positive_rev) + len(negative_rev)
data = np.zeros((xlen, len(word_map_index1)+1))

j = 0
for wordss in positive_rev:
    data[j:] = dict_to_array(wordss, 1)
    j += 1

for wordss in negative_rev:  
    data[j:] = dict_to_array(wordss, 0)
    j += 1


np.random.shuffle(data)

x_test = data[:-1000,:-1]
x_train = data[-1000:,:-1]

y_test = data[:-1000, -1]
y_train =data[-1000: , -1]

''' let's make the model'''
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(metrics.accuracy_score(y_test,y_pred))
import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
#from sklearn import LogisticRegrssion
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
    for x in stri:
        token = x.lower()
        token = nltk.tokenize.word_tokenize(stri)
        token = [x for x in token if x not in stopwords]
        token = [lemmatizer.lemmatize(x) for x in token]
        token = [x for x in token if len(x) > 3]
    return token


for x in negative_soup:
    negative_token = tokeniz(x.text)

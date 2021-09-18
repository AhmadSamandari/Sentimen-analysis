import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
#from sklearn import LogisticRegrssion
from nltk import word_tokenize

#######

stopwords = set([x.rstrip() for x in open('stopwords.txt')])


positive_soup = BeautifulSoup(
    open('positive.review').read(),  features="lxml")
positive_soup = positive_soup.findAll('review_text')
negative_soup = BeautifulSoup(
    open('negative.review').read(),  features="lxml")
negative_soup = negative_soup.findAll('review_text')

# print(positive_soup)

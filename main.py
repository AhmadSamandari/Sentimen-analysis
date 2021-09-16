import nltk
import numpy as numpy
from nltk import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

Lemmatizer = WordNetLemmatizer()  # first we need to initialize

# make a set of stopwords from the words in text
stopwords = set(word.rstrip() for word in open('stopwords.txt'))
# print(stopwords)

# with open(negative.review,  )

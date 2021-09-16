from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from nltk import WordNetLemmatizer
import numpy as np
import nltk
# nltk.download()

Lemmatizer = WordNetLemmatizer()  # first we need to initialize

# make a set of stopwords from the words in text
stopwords = set(word.rstrip() for word in open('stopwords.txt'))
# print(stopwords)

Positive_rev = BeautifulSoup(open('positive.review').read(), features="lxml")
Positive_rev = Positive_rev.findAll('review_text')
# print(Positive.prettify())
Negative_rev = BeautifulSoup(open('negative.review').read(), features='lxml')
Negative_rev = Negative_rev.findAll('review_text')
# print(Negative_rev)

np.random.shuffle(Positive_rev)  # to shuffle the comments
# to have equal number of + nd -
Positive_rev = Positive_rev[:len(Negative_rev)]


def token_maker(strings):
    strings.lower()  # to make them all lowercase
    words = nltk.tokenize.word_tokenize(strings)
    words = [t for t in words if len(t) >= 2]  # remove short words
    words = [Lemmatizer.lemmatize(t) for t in words]
    words = [t for t in words if t not in stopwords]  # remove stopwords
    return words


positive_tokens = []
negative_tokens = []

word_index_map = {}
# it increased when we have new word to find the vocabulary size
current_index = 0

for review in Positive_rev:  # to itrate in each review
    # We give each sentence to the function so produce the tokens
    tokens = token_maker(review.text)
    positive_tokens.append(tokens)
    for t in tokens:
        if t not in word_index_map:
            word_index_map[t] = current_index
            current_index += 1


for review in Negative_rev:  # to itrate in each review
    # We give each sentence to the function so produce the tokens
    tokens = token_maker(review.text)
    negative_tokens.append(tokens)
    for t in tokens:
        if t not in word_index_map:
            word_index_map[t] = current_index
            current_index += 1

# print(word_index_map)

# Now we want to make data array


def token_to_vector(token, label):
    x = np.zeros(len(word_index_map) + 1)  # 1 is for label
    for t in token:
        i = word_index_map[t]
        x[i] += 1
    x = x/x.sum()  # in order to normaliaze it
    x[-1] = label
    return x


''' MACHINE
        LEARNING
                PART'''

N = len(positive_tokens) + len(negative_tokens)
data = np.zeros((N, len(word_index_map) + 1))
i = 0

for token in positive_tokens:
    xy = token_to_vector(token, 1)
    data[i:] = xy
    i += 1

for token in negative_tokens:
    xy = token_to_vector(token, 0)
    data[i:] = xy
    i += 1


np.random.shuffle(data)

print(len(data))
X = data[:, :-1]
Y = data[:, -1]

x_train = X[:-100, ]
y_train = Y[:-100, ]

x_test = X[-100:, ]
y_test = Y[-100:, ]

model = LogisticRegression()
model.fit(x_train, y_train)
s = model.score(x_test, y_test)
print("classificatipon rate:{}".format(s))


threshold = 0.5
for word, index in word_index_map.iteritem():
    weight = model.coef_[0][index]
    if weight > threshold or weight < threshold:
        print(word, weight)

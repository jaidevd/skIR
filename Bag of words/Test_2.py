from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

vectorizer=TfidfVectorizer()
corpus = []

def fitting():
    global corpus
    text=open('newfile.txt')
    corpus = text.readlines()
    text.close()
    X = vectorizer.fit_transform(corpus)
    Y=X.toarray()
    return Y
    #print(Y)

def transform(sample1):
    X = vectorizer.transform([sample1])
    #print(X)
    print(cosine_similarity(X,Y).max())
    p=cosine_similarity(X,Y).ravel().argmax()
    #print(p+1)
    print(corpus[p])

Y=fitting()
while True:
    sample1=input("Enter your query: ")
    if sample1=='0':
        break;
    transform(sample1)

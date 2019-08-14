import time
import json
import tornado
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

vectorizer=TfidfVectorizer()
text=open('newfile.txt')
corpus = text.readlines()
text.close()
X = vectorizer.fit_transform(corpus)
Y=X.toarray()

def transform(sample1):
    X = vectorizer.transform([sample1])
    p=cosine_similarity(X,Y).ravel().argmax()
    return corpus[p]

def Query(handler):
    query= handler.get_argument("query")
    #handler.set_header('Content-Disposition'",value="k")
    
    return transform(query)

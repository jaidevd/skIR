import tornado
import requests
import numpy as np
import elasticsearch as es
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

vectorizer=TfidfVectorizer()
e=es.Elasticsearch()
c=es.client.CatClient(e)
i=es.client.IndicesClient(e)
dict={}

listI=c.indices(h="index").split("\n")
corpus=[]
for ind in listI:
    if ind=='':
        continue
    mappings=i.get_mapping(index=ind)
    columns= mappings[ind]["mappings"]["properties"].keys()

    keysAsList= list(columns)

    corpus=corpus+keysAsList
    dict[ind]=list(keysAsList)

X = vectorizer.fit_transform(corpus)
Y = X.toarray()

def transform(sample1):
    X = vectorizer.transform([sample1])
    p=cosine_similarity(X,Y).ravel().argmax()
    word=corpus[p]
    indexname=""
    for (key,value) in dict.items():
        if word in value:
            indexname=key
    return ("IndexName :"+indexname+" Column:"+word)
def Query(handler):
    query= handler.get_argument("query")
    return transform(query)


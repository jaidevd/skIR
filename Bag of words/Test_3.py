

import tornado  

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
vectorizer=TfidfVectorizer()
file_list = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f)) and f.endswith("csv")]

fcmap={}
columnf= open("columns","w+")
for f in file_list:
    csvf = open(f)
    columnl=csvf.__next__().split(",")
    fcmap[f]=columnl
    for word in columnl:
        columnf.write(word+"\n")
    csvf.close()

columnf.close()
columnf=open("columns")
corpus = columnf.readlines()
X = vectorizer.fit_transform(corpus)
Y = X.toarray()




def transform(sample1):
    X = vectorizer.transform([sample1])
    p=cosine_similarity(X,Y).ravel().argmax()
    word=corpus[p][:-1]
    filename=""
    for f in fcmap:
        columnl=fcmap[f]
        for column in columnl:
            if(word==column):
                filename=f
    return ("File : "+filename+" Column : "+corpus[p])
def Query(handler):
    query= handler.get_argument("query")
    #handler.set_header('Content-Disposition'",value="k")
    return transform(query)
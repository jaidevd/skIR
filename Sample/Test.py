from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

vectorizer=TfidfVectorizer()
corpus = []
text=open('newfile.txt')

for line in text:
    corpus.append(line)

X = vectorizer.fit_transform(corpus)

Y=X.toarray()
#print(Y)

sample1=input("Enter your query: ")
X = vectorizer.transform([sample1])
#print(X)


print(cosine_similarity(X,Y).max())
print(1+int(cosine_similarity(X,Y).argmax()))

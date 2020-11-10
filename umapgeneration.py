import pandas as pd
import numpy as np
import umap
from sklearn.feature_extraction.text import TfidfVectorizer

covidtrials= pd.read_csv("C:\clustering\\allresults.csv")

def TFIDF(X_train, MAX_NB_WORDS=75000):
    vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    print("tf-idf with",str(np.array(X_train).shape[1]),"features")
    columns=vectorizer_x.get_feature_names()
    return (X_train, columns)


Y=covidtrials['newc']
vectorizer = TfidfVectorizer(max_features=75000)
X = vectorizer.fit_transform(Y)

YY = covidtrials['clusters']

mapper = umap.UMAP().fit(X)

import umap.plot

p = umap.plot.points(mapper, labels=YY, color_key_cmap='Paired')

umap.plot.plt.show()

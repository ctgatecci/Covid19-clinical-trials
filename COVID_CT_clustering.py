import pandas as pd
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
import matplotlib.cm as cm
import heapq




allrecord = pd.read_csv("C:\clustering\ie_parsed_clinical_trials-new.csv")
allrecord['eligibility_type_new'] = allrecord['eligibility_type'].str[:3]
allrecord = allrecord.astype(str)
print(allrecord.shape)

allrecord = allrecord.drop(allrecord[allrecord.concepts =="nan"].index)
print(allrecord.shape)

allrecord2 = allrecord.apply(lambda x: x.str.strip() if x.dtype == "object" else x)


spec_chars = [", "]
for char in spec_chars:
    allrecord2['concepts'] = allrecord2['concepts'].str.replace(char, ' ')

spec_chars = ["(",")"]
for char in spec_chars:
    allrecord2['concepts'] = allrecord2['concepts'].str.replace(char, ' ')

spec_chars = [" "]
for char in spec_chars:
    allrecord2['concepts'] = allrecord2['concepts'].str.replace(char, '_')

allrecord2['conceptsnew'] = allrecord2[['eligibility_type_new', 'concepts']].apply(lambda x: '_'.join(x[x.notnull()]), axis = 1)
newrecord = allrecord2[['#nct_id','eligibility_type','conceptsnew']]

#newrecord.to_csv("C:\clustering\inclusionexclusion.csv") save to new file


output_series = newrecord.groupby(['#nct_id'])['conceptsnew'].apply(','.join).reset_index()
output_series['newconcepts'] = (output_series['conceptsnew'].str.split(',')
                              .apply(lambda x: OrderedDict.fromkeys(x).keys())
                              .str.join(','))

#output_series.to_csv("C:\clustering\preprocesseddata.csv") save the preprocessed data




def TFIDF(X_train, MAX_NB_WORDS=75000):
    vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    print("tf-idf with",str(np.array(X_train).shape[1]),"features")
    columns=vectorizer_x.get_feature_names()
    return (X_train, columns)


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


Y=output_series['newconcepts']
vectorizer = TfidfVectorizer(max_features=75000)
X = vectorizer.fit_transform(Y)

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from numpy import concatenate


n_clusters=11
random_state = 5
''' using silhouette value to determine the number of clusters. Please uncomment this part to try various n_clusters values. We've tried n_clusters up to 50
needtobreak=0
for n in range(2,n_clusters):
    clf = KMeans(n_clusters=n, random_state=random_state)
    data = clf.fit(X)
    YY= clf.labels_
    centroids = clf.cluster_centers_
    silhouette_avg = silhouette_score(X, YY)
    sample_silhouette_values = silhouette_samples(X, YY)
    cluster_silhouette_scores = []  # silhouette score for each cluster
    for i in range(n):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[YY == i]
        cluster_silhouette_scores.append(np.mean(ith_cluster_silhouette_values))
        #if (np.mean(ith_cluster_silhouette_values) ==0):
        print(ith_cluster_silhouette_values)
        print(len(ith_cluster_silhouette_values))
        if len(ith_cluster_silhouette_values)<20:
            needtobreak=1;
            break;

    print(cluster_silhouette_scores)
    print("For n_clusters =", n,
          "The average silhouette_score is :", silhouette_avg)
    filename = "Kmeanssilall.csv"
    f = open(filename, 'a')
    f.write(str(silhouette_avg))
    f.write('\n')
    f.close()
    if needtobreak>0:
        break;

'''

clf = KMeans(n_clusters=n_clusters, random_state=random_state)
data = clf.fit(X)
YY= clf.labels_
centroids = clf.cluster_centers_



frame =  pd.DataFrame()
clustercontent=pd.DataFrame()
frame['#nct_id'] = output_series['#nct_id']
frame['newc']=Y
frame['clusters']= clf.labels_
counteachc = frame.groupby(['clusters'])
countframe =counteachc.count()
frame.to_csv("C:\clustering\\allresults.csv") #save clustering results
order_centroids = clf.cluster_centers_.argsort()[:, ::-1]
labels = ['']*n_clusters
labelsfreq = ['']*n_clusters
for i in range(n_clusters):
    print("Cluster %d words:" % i, end='')
    clustercontent = frame.loc[frame['clusters'] == i]
    wordfreq = {}
    for sentence in clustercontent['newc']:
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1
    wordfreq.pop(',', None)
    frequencyword= pd.DataFrame.from_dict(wordfreq, orient='index').reset_index()
    frequencyword.columns=['concept', 'freq']
    filename = "C:\clustering\clusterid"+str(i)+"cfreq.csv"
    frequencyword.to_csv(filename)
    most_freq = heapq.nlargest(10, wordfreq, key=wordfreq.get) #get the top 10 frequent terms
    if (most_freq.count(',')>0):
        most_freq.remove(',')
    labelsfreq[i] = ','.join(most_freq)
    labels[i]="cluster"+str(i)

clusterlabel=pd.DataFrame(labelsfreq, columns=['clusterlabel'])
clusterlabel['clusterid'] = range(0,n_clusters)
clusterlabel['nctcounts']=countframe['#nct_id']
clusterlabel.to_csv("C:\clustering\clusterlabelscounts.csv") #save label frequency

plt.figure(figsize=(10, 10))
everything = concatenate((X.todense(), centroids))
tsne_init = 'pca'
tsne_perplexity = 20.0
tsne_early_exaggeration = 4.0
tsne_learning_rate = 10
model = TSNE(n_components=2, random_state=random_state, init=tsne_init,
    perplexity=tsne_perplexity,
    early_exaggeration=tsne_early_exaggeration, learning_rate=tsne_learning_rate)

transformed_everything = model.fit_transform(everything)
print(transformed_everything)
colors = cm.rainbow(np.linspace(0, 1, n_clusters))

for l, c, co, in zip(labels, colors, range(n_clusters)):
    plt.scatter(transformed_everything[np.where(YY == co), 0],
                transformed_everything[np.where(YY == co), 1],
                marker='o',
                color=c,
                linewidth='1',
                alpha=0.8,
                label=l)


plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), prop={'size': 10})
plt.savefig("visualization-NCT-COVID-thetha20-v1.png",bbox_inches='tight', dpi=300)
plt.show()

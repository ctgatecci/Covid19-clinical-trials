import pandas as pd
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import sklearn.preprocessing as skp
import nltk
from sklearn.cluster import KMeans
import heapq


covidtrials= pd.read_csv("C:\clustering\COVID-19_trials_with_study_design.csv")
cvenrollment = covidtrials[['id','enrollment','intervention_type','study_type']]

allrecord = pd.read_csv("C:\clustering\ie_parsed_clinical_trials_10122020.csv")
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


output_all = newrecord.groupby(['#nct_id'])['conceptsnew'].apply(','.join).reset_index()
output_all['newconcepts'] = (output_all['conceptsnew'].str.split(',')
                              .apply(lambda x: OrderedDict.fromkeys(x).keys())
                              .str.join(','))


#output_series.to_csv("C:\clustering\preprocesseddata.csv") save the preprocessed data


output_series = output_all.join(cvenrollment.set_index('id'), on='#nct_id')

output_series = output_series[output_series.enrollment != 0] # remove the studies that have 0 enrollment

output_series = output_series[output_series.study_type == 'Interventional'] #only consider the interventional study when include enrollment as a dimension

output_series.to_csv("C:\clustering\preprocesseddata.csv") #save the preprocessed data


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


'''extract the enrollment and normalize the value'''
w=output_series[['enrollment']].values.astype(float)
min_max_scaler = skp.MinMaxScaler()
w_scaled = min_max_scaler.fit_transform(w)
enroll = pd.DataFrame(data = w_scaled
             , columns = ['w'])

'''use PCA to reduce the dimension to 10 '''
from sklearn.decomposition import SparsePCA
pca = SparsePCA(n_components=10)
principalComponents = pca.fit_transform(X.toarray())
Xdf = pd.DataFrame(data = principalComponents
             , columns = ['pc1', 'pc2', 'pc3','pc4','pc5','pc6', 'pc7', 'pc8','pc9','pc10'])


finalX = pd.concat([Xdf, enroll], axis = 1)  #adding the enrollment as 11th dimension



n_clusters=8
random_state = 5

'''
 
 #using silhouette, DBindex, CHindex and SSE to determine the number of clusters. 
 #Please uncomment this part to try various n_clusters values. We've tried n_clusters up to 50, and n_clusters=8 was selected
 
n_clusters=50
sse = {}
dunn ={}
for n in range(2,n_clusters):
    clf = KMeans(n_clusters=n, max_iter=10000, random_state = random_state)
    #data = clf.fit(X,sample_weight=w)
    data=clf.fit(finalX)
    YY= clf.labels_
    chscore = calinski_harabasz_score(X.toarray(), YY)
    dbindx= davies_bouldin_score(X.toarray(), YY)
    centroids = clf.cluster_centers_
    silhouette_avg = silhouette_score(X.toarray(), YY)
    sample_silhouette_values = silhouette_samples(X, YY)
    cluster_silhouette_scores = []  # silhouette score for each cluster
    sse[n-1] = clf.inertia_

    for i in range(n):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[YY == i]
        cluster_silhouette_scores.append(np.mean(ith_cluster_silhouette_values))


    print(cluster_silhouette_scores)
    print("For n_clusters =", n,
          "The average silhouette_score is :", silhouette_avg)
    filename = "KMsilall-rn0-unw-enroll-pca10-in1.csv"
    f = open(filename, 'a')
    f.write(str(silhouette_avg))
    f.write('\n')
    f.close()

    filename = "KMDBindexall-rn0-unw-enroll-pca10-in1.csv"
    f = open(filename, 'a')
    f.write(str(dbindx))
    f.write('\n')
    f.close()

    filename = "KMCHScoreall-rn0-unw-enroll-pca10-in1.csv"
    f = open(filename, 'a')
    f.write(str(chscore))
    f.write('\n')
    f.close()

plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()
'''
clf = KMeans(n_clusters=n_clusters,random_state = random_state)
data=clf.fit(finalX)
YY= clf.labels_
centroids = clf.cluster_centers_
cluster_silhouette_scores = []
sample_silhouette_values = silhouette_samples(X, YY)
for i in range(n_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[YY == i]
    cluster_silhouette_scores.append(np.mean(ith_cluster_silhouette_values))

frame =  pd.DataFrame()
clustercontent=pd.DataFrame()
frame['#nct_id'] = output_series['#nct_id']
frame['newc']=Y
frame['clusters']= clf.labels_
frame['enrollment']=output_series['enrollment']

counteachc = frame.groupby(['clusters'])
sumframe= counteachc.sum()
countframe =counteachc.count()
print(countframe)
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
clusterlabel['totalenrollment'] = sumframe['enrollment']
clusterlabel['cluster_silhouette_scores']=cluster_silhouette_scores
clusterlabel.to_csv("C:\clustering\clusterlabelscounts.csv") #save label frequency



import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics


df = pd.read_json("Brisbane_CityBike.json")
print("number of null data",df.isnull().sum())
X=df.loc[:,['latitude','longitude']]

kmax=10
mylist = []

for i in range(2,kmax):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
    id_label=kmeans.labels_
    coefficient = metrics.silhouette_score(X, kmeans.labels_)
    print('pour k= ',i)
    print('Silhouette coefficient: {:0.03f}'.format(metrics.silhouette_score(X, kmeans.labels_))) 
    mylist.append(coefficient)
arr=np.asarray(mylist)
#idx = (-arr).argsort()[:2]


symbols = np.array(['b.','r.','m.','g.','c.','k.','b*','r*','m*','r^']);
plt.figure(figsize=(7,7))
plt.ylabel('Longitude', fontsize=10)
plt.xlabel('Latitude', fontsize=10)
k=6
kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
id_label=kmeans.labels_
df['cluster'] = kmeans.predict(X).tolist() 
for i in range(k):	  
    cluster=np.where(id_label==i)[0]
    plt.plot(X.latitude[cluster].values,X.longitude[cluster].values,symbols[i])

plt.title(" KMeans (k=6)")
plt.show()
df.to_csv('clustering_result.csv',sep=';')






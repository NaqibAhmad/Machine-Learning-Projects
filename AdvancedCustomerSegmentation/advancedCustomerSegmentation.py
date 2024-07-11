import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from kmodes.kprototypes import KPrototypes

df = pd.read_csv("C:/Users/Naqib Ahmad/Downloads/segmentation-data.csv")
#print(df.head())

#before scaling we keep normal values in temp variables 
df_temp = df[['ID','Age','Income']]
#print(df_temp)

#scaling the data
scaler = MinMaxScaler()
scaler.fit(df[['Age']])
df['Age']= scaler.transform(df[["Age"]])

scaler.fit(df[["Income"]])
df['Income'] = scaler.transform(df[['Income']])

#dropping ID as it is not used
df= df.drop(['ID'], axis= 1)
print(df.head(5))
#converting age and income into flaot
mark_array= df.values

mark_array[:, 2]= mark_array[:, 2].astype(float)
mark_array[:, 4]= mark_array[:, 4].astype(float)
print(df.head())

Kproto= KPrototypes(n_clusters=10, verbose=2, max_iter= 20)
clusters= Kproto.fit_predict(mark_array, categorical= [0,1,3,5,6])
print(Kproto.cluster_centroids_)
print(len(Kproto.cluster_centroids_))

clusters_col = []
for i in clusters:
    clusters_col.append(i)
    
df['cluster']= clusters_col
#putting original columns from temp to df
df[['ID', 'Age', 'Income']] = df_temp

print(df[df['cluster']==1].head(10))

#plotting the segmentation in graph
colors = ['green', 'red', 'gray', 'orange', 'yellow', 'cyan','magenta','brown','purple','blue']
plt.figure(figsize=(15,15))
plt.xlabel('Age')
plt.ylabel('Income')

for i, col in zip(range(10), colors):
    dftemp= df[df.cluster==i]
    plt.scatter(dftemp.Age, dftemp['Income'], color=col, alpha=0.5)
    
plt.legend()
plt.show()
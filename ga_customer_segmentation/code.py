# --------------
# import packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 



# Load Offers
offers = pd.read_excel(path, sheet_name = 0)

# Load Transactions
transactions = pd.read_excel(path, sheet_name = 1)

# Merge dataframes
df = pd.concat([offers, transactions], axis = 1)
transactions['n'] = 1
# Look at the first 5 rows
df.head(5)
print(df.shape)


# --------------
# Code starts here

# create pivot table
matrix = df.pivot_table(index='Customer Last Name', columns='Offer #', values='n')

# replace missing values with 0
matrix.fillna(0, inplace=True)

# reindex pivot table
matrix.reset_index(inplace=True)

# display first 5 rows
print(matrix.head())

# Code ends here


# --------------
# import packages
from sklearn.cluster import KMeans

# Code starts here

# initialize KMeans object
cluster = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

# create 'cluster' column
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[1:]])

print(matrix.head(5))
# Code ends here


# --------------
# import packages
from sklearn.decomposition import PCA

# Code starts here

# initialize pca object with 2 components
pca = PCA(n_components = 2, random_state = 0)

# create 'x' and 'y' columns donoting observation locations in decomposed form
matrix['x'] = pca.fit_transform(matrix[matrix.columns[1:]])[:,0]
matrix['y'] = pca.fit_transform(matrix[matrix.columns[1:]])[:,1]

# dataframe to visualize clusters by customer names
cluster = matrix.iloc[[0,33,34,35]]

# visualize clusters
cluster.plot.scatter(x='x', y='y', c='cluster')

# Code ends here


# --------------
# Code starts here

# merge 'clusters' and 'transactions'
data = pd.merge(clusters, transactions)

# merge `data` and `offers`
data = pd.merge(offers, data)
# initialzie empty dictionary
champagne = {}
n_clusters = data['cluster'].unique()

print(data.head(3), n_clusters, sep='\n')

# iterate over every cluster
for i in n_clusters:

    # observation falls in that cluster
    new_df = data[data['cluster'] == i]

    # sort cluster according to type of 'Varietal'
    counts = new_df['Varietal'].value_counts(ascending = False)

    # check if 'Champagne' is ordered mostly
    if(counts.index[0] == 'Champagne'):
        # add it to 'champagne'
        champagne[i] = counts[0]


# get cluster with maximum orders of 'Champagne' 
print(np.argmax(champagne.values()))

# print out cluster number




# --------------
# Code starts here

# empty dictionary
discount = {}

# iterate over cluster numbers
n_clusters = data['cluster'].unique()
print(data.head())
for i in n_clusters:
    # dataframe for every cluster
    new_df = data[data['cluster'] == i]
    # average discount for cluster
    counts = new_df['Discount (%)'].sum()/new_df.shape[0]

    # adding cluster number as key and average discount as value 
    discount[i] = counts

# cluster with maximum average discount
cluster_discount = list(discount.keys())[np.argmax(list(discount.values()))]
print(discount, cluster_discount, sep = '\n')
# Code ends here



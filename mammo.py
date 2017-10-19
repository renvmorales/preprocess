

import pandas as pd 
import numpy as np



# mammographic dataset url 
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data'

# column names to be placed on the dataframe 
col_names = ['BI-RADS','Age','Shape','Margin','Density','Severity']



print('\nFirst 10 observations of dataset:')
df = pd.read_csv(url, header=None, names=col_names, na_values=['?'])
print(df.head(10))




#######################################################
# general information on dataset

# list of all attribute names
print('\nColumn names:\n', list(df.columns.values))

# total number of rows and columns 
print('\nNumber of rows: ', df.shape[0])
print('Number of columns: ', df.shape[1])


# total missing values
print('\nTotal missing values: ', df.isnull().values.sum())


# total missing values per column
print('\nTotal missing values per column:')
print(df.isnull().sum().to_string(index=True))



# percentual missing values per column
print('\nPercentual missing values per column:')
print((df.isnull().sum()/df.shape[0]*100).to_string(index=True))



######################################################

# impute missing values using the median of the same output class
for i in range(len(df.columns)):
	index = [ind for ind, j in enumerate(list(df.iloc[:,i].values)) if np.isnan(j)]
	for k in index:
		med = df[df.iloc[:,-1]==df.iloc[k,-1]].iloc[:,i].median()
		df.iloc[k,i] = med


print('\nImputing missing values ...')



print('\nTotal missing values: ', df.isnull().values.sum())

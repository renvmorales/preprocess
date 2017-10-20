

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# mammographic dataset url 
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data'

# column names to be placed on the dataframe 
col_names = ['BI-RADS','Age','Shape','Margin','Density','Severity']



print('\nFirst 10 observations of dataset:')
df = pd.read_csv(url, header=None, names=col_names, na_values=['?'])
print(df.head(10))




#######################################################
# general information on  (total missing values, number of affected objects...)
# check also consistency of data (range value, categorical limits...)
# plot histograms for evey column


# list of all attribute names
print('\nColumn names:\n', list(df.columns.values))

# total number of rows and columns 
print('\nNumber of rows: ', df.shape[0])
print('Number of columns: ', df.shape[1])




# total missing values per column
print('\nTotal missing values per column:')
print(df.isnull().sum().to_string(index=True))


# percentual missing values per column
print('\nPercentual missing values per column:')
print((df.isnull().sum()/df.shape[0]*100).to_string(index=True))


# total missing values
print('\nTotal missing values: ', df.isnull().values.sum())
# number of objects with at least one missing value
print('Affected number of objects: ', df.isnull().any(axis=1).sum())



# identify limits and all possible types of values loop
for i in range(len(df.columns)):
	# min and max for BI-RADS column
	print("\n[Min - max] interval for '%s' column: [%d - %d]" % (df.columns[i], df.iloc[:,i].min(), df.iloc[:,i].max()))
	numtypes = pd.unique(df.iloc[:,i])
	if len(numtypes) < 20:
		print('Different types values: ', numtypes)
	else:
		print('Number of types exceed 20 or variable might be continuous.')
	print('Plotting histogram ...')
	g = sns.distplot(df.iloc[:,i].dropna(), label=df.columns[i], kde=False)
	plt.legend()
	plt.show()



# # exclude out of expected range values
# df.iloc[df.iloc[:,0]==55, 0] = np.nan
# df.iloc[df.iloc[:,0]==0, 0] = np.nan
# df.iloc[df.iloc[:,0]==6, 0] = np.nan

print('Number of different elements for BI-RADS column: ', pd.unique(df['BI-RADS']))



######################################################
# missing values imputation on the dataset  

# impute missing values using the median of the same output class
for i in range(len(df.columns)):
	index = [ind for ind, j in enumerate(list(df.iloc[:,i].values)) if np.isnan(j)]
	for k in index:
		med = df[df.iloc[:,-1]==df.iloc[k,-1]].iloc[:,i].median()
		df.iloc[k,i] = med


print('\nImputing missing values ...')

# check if there is any missing value left
print('Total missing values: ', df.isnull().values.sum())





######################################################################
# dataset normalization 

print('Number of different elements per column: ',pd.unique(df['BI-RADS']))


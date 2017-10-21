

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# mammographic dataset url 
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data'

# column names to be placed on the dataframe 
col_names = ['BI-RADS','AGE','SHAPE','MARGIN','DENSITY','SEVERITY']



print('\nFirst 10 observations of dataset:')
df = pd.read_csv(url, header=None, names=col_names, na_values=['?'])
print(df.head(10))




#######################################################
# general information on  (total missing values, number of affected objects...)

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





######################################################################
# check consistency of data (range value, categorical limits...)
# plot histograms for evey column

print('\n>>>>Checking consistency limits ...')

# identify limits and all possible types of values loop
for i in range(len(df.columns)):
	# min and max for BI-RADS column
	print("\n[Min - max] interval for '%s' column: [%d - %d]" % (df.columns[i], df.iloc[:,i].min(), df.iloc[:,i].max()))
	numtypes = np.sort(pd.unique(df.iloc[:,i]))
	if len(numtypes) < 20:
		print('Different types values: ', numtypes)
	else:
		print('Number of types exceed 20 or variable might be continuous.')
	print('Plotting histogram ...')
	g = sns.distplot(df.iloc[:,i].dropna(), label=df.columns[i], kde=False)
	plt.legend()
	plt.show()






################################################################
# manually exclude out-of-expected range values

print('\n>>>>Excluding and correcting out-of-range values ...')

df.loc[df.loc[:,'BI-RADS']==55, 'BI-RADS'] = 5
df.loc[df.loc[:,'BI-RADS']==0, 'BI-RADS'] = 5
df.loc[df.loc[:,'BI-RADS']==6, 'BI-RADS'] = np.nan






######################################################
# missing values imputation on the dataset  

print('\n>>>>Imputing missing values ...')

# impute missing values using the median of the same output class
for i in range(len(df.columns)):
	index = [ind for ind, j in enumerate(list(df.iloc[:,i].values)) if np.isnan(j)]
	for k in index:
		med = df[df.iloc[:,-1]==df.iloc[k,-1]].iloc[:,i].median()
		df.iloc[k,i] = med


# check if there is any missing value left
print('Total missing values after imputation: ', df.isnull().values.sum())






######################################################################
# discretize 'Age' column values with fixed bin size

print("\n>>>>Discretizing 'AGE' column ...")

Age_max = df['AGE'].max() # max age value
Age_min = df['AGE'].min() # min age value
n_bins = 6  # selected number of bins


# every pair of points defines the limits of the box
limits = np.arange(Age_min, Age_max+0.1, (Age_max-Age_min)/n_bins)


# centroid of every bin represents all values inside 
ds_mean = []
for i in range(len(limits)-1):
	ds_mean.append((limits[i+1]+limits[i])/2)

ds_mean = np.floor(np.array(ds_mean))  # reduce to integer values
limits = limits[1:]


# define function to discretize
def get_centroid(x, limits, ds_mean):
	ind = list(x<=limits).index(True)
	return ds_mean[ind]

# vectorized discretization
df['AGE'] = df['AGE'].apply(get_centroid, args=(limits, ds_mean) )


print('Different types values after imputation', 
	np.sort(pd.unique(df['AGE'])) )






######################################################################
# dataset max-min normalization 

print('\n>>>>Normalizing dataset ...')
df = df.apply(lambda x: (x-x.min())/(x.max()-x.min()))






######################################################################
# display final preprocessed dataset

print('\nSample of preprocessed final dataset:')
print(df.applymap(lambda x: '%.2f' % x).head(10))





#########################################################################
# convert data to array and save as file

print('\n>>>>Converting dataset to array and saving to file ...')
np.savez('mammo_df', 
	X=df.iloc[:,1:-1].as_matrix(), 
	Y=df.iloc[:,-1].as_matrix(), 
	BIRADS=df.iloc[:,0].as_matrix())




print('\nPreprocessing data complete.')
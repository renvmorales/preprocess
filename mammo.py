

import pandas as pd 



# mammographic dataset url 
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data'

# column names to be placed on the dataframe 
col_names = ['BI-RADS','Age','Shape','Margin','Density','Severity']



print('\nFirst 10 observations of dataset:')
df = pd.read_csv(url, header=None, names=col_names)
print(df.head(10))




#######################################################
# general information on dataset

# list of all attribute names
print('\nColumn names:\n', list(df.columns.values))

# total number of rows and columns 
print('\nNumber of rows: ', df.shape[0])
print('Number of columns: ', df.shape[1])


# total missing values
print('\nTotal missing values: ', (df.values=='?').sum())


# total missing values per column
print('\nTotal missing values per column:')
print(df.applymap(lambda x: x=='?').sum(axis=0).to_string(index=True))




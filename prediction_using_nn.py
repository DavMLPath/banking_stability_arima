import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

z_score = pd.read_excel('C:/Users/37493/Downloads/Factors.xlsx', sheet_name="Z-score")
z_score.set_index('Month', inplace=True)
z_score.drop(['Unnamed: 2', 'Unnamed: 3'],axis=1, inplace=True)
print(z_score.head(10))

y=z_score
X=pd.read_excel('C:/Users/37493/Downloads/Factors.xlsx', sheet_name='Factors')
print('_________this is first 10 rows of x________')
print(X.head(10))
print('_________this is info________')
print(X.info())
print('_________this is describe________')
print(X.describe())

X=X.replace('-',0)
X=X.replace('####',0)
print(X.columns.values)
X.rename(columns = {' ': 'Month'}, inplace = True)
X.index=X['Month']
#print(X.head(10))
X=X[X.index>='2012-01-01']
print('here are indexes',X.index.values)


#print(X.loc['Month',1])
#
# print('a analyses starts from here---------')
#
# #print(type(X.iloc[:,22]))
# #print('this is sum of first 5 elems',X.iloc[:,22][0:5].sum())
# print('This Are  series',
#       X.iloc[:, 22][8 - 2:8 + 3] )
# print('this is sum of series',X.iloc[:,22][8-2:8+3].sum())
# print('this is number of  non zeros',(X.iloc[:,22][8-2:8+3]!=0).count())
#
# #print('This is a',a)
# #print('this is sym of a',sum(a))
#
# #
# # print(X.iloc[:, 22][X.iloc[:, 22].isnull()])
# # print(X.iloc[:, 22].count())
#
#
#
#
print(X.iloc[5:6,2])
X=X.drop('Month', axis=1)
X=X.reset_index(drop=True)
print(X.head(10))

for i in X:
    for j in range(len(X[i])):
       # print(X[i][j])
        if pd.isna(X[i][j]):
            X.loc[j,i]=(X[i][j+1]+X[i][j-1])/2
print(X.iloc[100:107,21])



for i in X:
    for j in range(len(X.iloc[:,0])):
         if X.loc[j,i]==0:
        #         print(X.loc[i,j])
            X.loc[j,i]=(X.loc[j-2:j+3,i]).sum()/((X.loc[j-2:j+3,i]!=0).count()-1)
            print(X[i][j])
            print(X.loc[j - 2:j + 3, i])
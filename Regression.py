import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
y=pd.read_excel(r'C:\Users\Davit\Desktop\Z-score recalculated.xlsx', sheet_name='Z-score')
# Load the data into a pandas dataframe

y.set_index('Month', inplace=True)
X=pd.read_excel(r'C:\Users\Davit\Downloads\Factors.xlsx', sheet_name='Factors')
y=y.iloc[:124, :]
X=X[(X['Month']>='2012-01-01') &  (X['Month']<='2022-04-01')]
X.set_index('Month', inplace=True)



#________________________________
# X=X.replace('-',0)
# X=X.replace('NaN',0)
#
# X=X.replace('####',0)
# X=X.replace('#######',0)
#---------------------------------------


#print(len(X ))
#print(len(y))
#factor_growth=pd.read_excel(r'C:\Users\Davit\Desktop\Factors_growth.xlsx', sheet_name='Factors')
#corr_m_growth=factor_growth.corr()
#corr_m_growth.to_excel(r'C:\Users\Davit\Desktop\corr_matrix_growth.xlsx', index=False)
# print(X.iloc[-1,:2] )
# print(y.iloc[-1,0:2])
#
#

X=X.replace('-',np.nan)

X=X.replace('####',np.nan)
X=X.replace('#######',np.nan)
X=X.fillna(method='ffill')
print(X.columns.values)
X=X.drop('Կառավարության ԶՈՒՏ ՆԵՐՔԻՆ ԱԿՏԻՎՆԵՐ', axis=1)

X = X.applymap(lambda i: np.log(i))
#X.to_excel(r'C:\Users\Davit\Desktop\log_data.xlsx', index=False)

# X_growth= X.pct_change() * 100
# X_growth=X_growth.round(2)

# print(X_growth.head(10))
# X_growth=X_growth.iloc[1:,:]
#
# print(X_growth.head(10))








for i in X:
    if i not in ('ԱՄՆ դոլար','ԵՎՐՈ', 'Ռուսական ռուբլի'):
        X[i]=X[i]-X[i].shift(1).fillna(0)
        print(i)



#
#
corr_m_diff1=X.corr()
#corr_m_diff1.to_excel(r'C:\Users\Davit\Desktop\corr_m_diff_2.xlsx', index=False)


#
# #y.to_csv(r'C:\Users\Davit\Desktop\my_data.csv', index=False)
#
# X=X.fillna(0)
#     # new=y[y.isnull()==True]
#     # y['Z-score']=y['Z-score'].fillna(0)
#     # y['Z-score'] = y['Z-score'].replace(np.nan, 0)
#     # y[y.isnull()==True]=y[y.isnull()==True].replace(np.nan, 0)
#     # y.fillna(0)
#
#     #
#     #
#
#     # Fit the multiple linear regression model
# model = sm.OLS(y,X)
# result=model.fit()
#     # # Print the summary of the model
# print('Here are the params',result.params)
# print('Here are the T values',result.tvalues)
# print(result.rsquared)
#
# corr_m=X.corr()
#
#
#
# coef=pd.DataFrame(result.params)
#
#
# print(type(result.params))
# print(result.summary())
#
# #coef.to_csv(r'C:\Users\Davit\Desktop\coefficients.csv', index=False)
# print(corr_m)
# print(type(corr_m))
# corr_m.to_csv(r'C:\Users\Davit\Desktop\corr_matrix_new.csv', index=False)
#
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sb
# dataplot = sb.heatmap(X.corr())
#
# a=pd.DataFrame(X.columns)
# #a.to_excel(r'C:\Users\Davit\Desktop\corr_matrix.xlsx', index=False)
#
# plt.show()
# print(a)
#
#
X=X.loc[:,['Մինչև 1 տարի ժամկետով Իրավաբանական անձանցից ավանդներ ՀՀ դրամ',
 'Մինչև 1 տարի ժամկետով Իրավաբանական անձանցից ավանդներ ԱՄՆ դոլար',
 '1 տարուց ավել ժամկետով Իրավաբանական անձանցից ավանդներ ՀՀ դրամ',
 '1 տարուց ավել ժամկետով Իրավաբանական անձանցից ավանդներ ԱՄՆ դոլար',
 'Ֆիզիկական անձանց հիփոթեքային վարկեր (ՀՀ դրամ)',
 'Պետական կարճաժամկետ պարտատոմսերի եկամտաբերությունը3',
 'Պետական միջինժամկետ պարտատոմսերի եկամտաբերությունը3',
 #'Կառավարության ԶՈՒՏ ՆԵՐՔԻՆ ԱԿՏԻՎՆԵՐ',
 'Բանկերի ԶՈՒՏ ՆԵՐՔԻՆ ԱԿՏԻՎՆԵՐ',
 'Թղթակցային հաշիվներ դրամով',
 'Թղթակցային հաշիվներ արտարժութով',
 'ԱՄՆ դոլար',
 'ԵՎՐՈ']]

#
# X_growth=X_growth.loc[:,['Մինչև 1 տարի ժամկետով Իրավաբանական անձանցից ավանդներ ՀՀ դրամ',
#  'Մինչև 1 տարի ժամկետով Իրավաբանական անձանցից ավանդներ ԱՄՆ դոլար',
#  '1 տարուց ավել ժամկետով Իրավաբանական անձանցից ավանդներ ՀՀ դրամ',
#  '1 տարուց ավել ժամկետով Իրավաբանական անձանցից ավանդներ ԱՄՆ դոլար',
#  'Ֆիզիկական անձանց հիփոթեքային վարկեր (ԱՄՆ դոլար)',
#  #'Պետական կարճաժամկետ պարտատոմսերի եկամտաբերությունը3',
# # 'Պետական միջինժամկետ պարտատոմսերի եկամտաբերությունը3',
#  'Կառավարության ԶՈՒՏ ՆԵՐՔԻՆ ԱԿՏԻՎՆԵՐ',
#  'Բանկերի ԶՈՒՏ ՆԵՐՔԻՆ ԱԿՏԻՎՆԵՐ',
#  'Թղթակցային հաշիվներ դրամով',
#  'Թղթակցային հաշիվներ արտարժութով',
#  'ԱՄՆ դոլար',
#  'ԵՎՐՈ']]
#
#
# for i in X_growth:
#     X_growth[i]=X_growth[i]-X_growth[i].shift(1).fillna(0)
#

#X_growth.to_excel(r'C:\Users\Davit\Desktop\growth_data2.xlsx', index=False)



#_____________________________________________________________________________________
# for i in X_growth:
#     #print('this is null value for {}'.format(i))
#     #print(X[i][X[i].isnull()])
#     X_growth[i]=X_growth[i].fillna(0)

#
# for i in X_growth:
#     print(X_growth[i][X_growth[i].isnull()])
#
# for i in X_growth:
#     print('this is adf fuller of {}'.format(i) )
#     result = adfuller(X_growth[i])
#     adf_statistic = result[0]
#     p_value = result[1]
#
#     print("ADF Statistic:", adf_statistic)
#     print("p-value:", p_value)
#     if p_value<=0.05:
#         print('this is stationary')
#     else: print('no ape jan not this time, non stationary ')
#
#



#
# #_____________________________________________________________________________________
# for i in X:
#     #print('this is null value for {}'.format(i))
#     #print(X[i][X[i].isnull()])
#     X[i]=X[i].fillna(0)

#
# for i in X:
#     print(X[i][X[i].isnull()])
#
# print(type(X))
#
for i in X:
    print('this is adf fuller of {}'.format(i) )
    result = adfuller(X[i])
    adf_statistic = result[0]
    p_value = result[1]

    print("ADF Statistic:", adf_statistic)
    print("p-value:", p_value)
    if p_value<=0.05:
        print('this is stationary')
    else: print('no ape jan not this time, non stationary ')
#______________________________________________________________________________________________--
# #print(X.loc[:,'Պետական միջինժամկետ պարտատոմսերի եկամտաբերությունը3'])
# # # model2 = sm.OLS(y,X)
# # # result2=model2.fit()
#
# # print(X.head(10))
# print(len(X))
# print(len(y))
#
#
# print(y.index)
X.index=y.index
# X=X.fillna(0)
# #Adding constant
y = y.applymap(lambda i: np.log(i))



exog = X
#defining and fitting the model
model = sm.tsa.ARIMA(y, order=(2, 0, 1), exog=exog)
results = model.fit()
print(results.summary())
#--------------------------------------------------------------------------
# y= y.pct_change() * 100
# y=y.iloc[1:, :]
# X_growth.index=y.index
#
# #Adding constant
# exog = X_growth
# #defining and fitting the model
# model = sm.tsa.ARIMA(y, order=(3, 1, 1), exog=exog)
# results = model.fit()
# print(results.summary())



















#f
#
# #
# # coef=pd.DataFrame(result2.params)
# #
# #
# # print(type(result2.params))
# # print(result2.summary())
# # print(result2.rsquared)
# #
# # from statsmodels.tsa.stattools import adfuller
#
# #Checking the stationarity of regressors
#
#
# #now lets make stationary each column elements
#
#
#
#
# #CHECK STATIONNARITY
# #adjustment, dif, log
# #choose order and optimal lag for ARIMA
#
# #lyung box q  test---showing autocorrelation
# #STABILITY CHECK
#
#
# #ARIMA (1,0) IS ARDEL
#
# #var MODEL-


y= y - y.shift(1).fillna(0)

result = adfuller(y)
adf_statistic = result[0]
p_value = result[1]
print("ADF Statistic:", adf_statistic)
print("p-value:", p_value)
if p_value<=0.05:
    print('this is stationary')
else: print('no ape jan not this time, non stationary ')





# #multifactor arima model
# import statsmodels.api as sm
#
# X=pd.read_excel(r'C:\Users\Davit\Downloads\Factors.xlsx', sheet_name='Factors')
# X=X.fillna(0)
# X=X.replace('-',0)
# X=X.replace('####',0)
# X=X.replace('#######',0)
# exog_data = X[(X['Month']>='2012-01-01') &  (X['Month']<='2022-04-01')]
#
#
# exog_data.set_index('Month', inplace=True)
#
# exog_train =exog_data[:int(0.8*(len(exog_data)))]
# exog_test=  exog_data[int(0.8*(len(exog_data))):]
# z_score_train=z_score[:int(0.8*(len(z_score)))]
# z_score_test=z_score[int(0.8*(len(z_score))):]
# print(len(exog_data))
# print(len(exog_train))
# print(len(z_score_train))
#
#
# result = exog_train.dtypes
#
# print("Output:")
# print(result)
#
#
# model = sm.tsa.statespace.SARIMAX(z_score_train, exog=exog_train, order=(24, 0, 1))
# results = model.fit()
# print(results.summary())
#
#
#
#
# forecast = results.forecast(steps=len(exog_test), exog=exog_test)
# print(forecast)
# print(len(forecast))
# print(len(z_score_test))
# print(forecast.head())
# print(z_score_test.head())
#
#
# print(type(forecast))
# print(type(z_score_test['Z-score']))
# import numpy as np
# df=pd.DataFrame({'forecasts':forecast, 'Z_score':z_score_test['Z-score']})
# print(df)
#
#
# opt_mse = np.mean((df['forecasts']-df['Z_score'])** 2)
# opt_rmse = np.sqrt(opt_mse)
# print(opt_mse)
#

#
# #Pur goal is to refactor this code, so that
# 1)We have clear code for regression
# 2)We have train test split
# 3) We have ARIMA single factor p,d,q found manually and with auto ARIMA, and compare results
# 4) We have meaningful results
# 5) we evaluate model and publish



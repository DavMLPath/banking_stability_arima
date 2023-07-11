import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from datetime import date, timedelta


z_score = pd.read_excel('C:/Users/Davit/Desktop/Z-score based on banking balance.xlsx', sheet_name="Z-score")
print(z_score.head())

z_score.set_index('Month', inplace=True)
print('here is the z-score', z_score.tail(10))


#Plotting the data
plt.plot( z_score['Z-score'])
plt.xlabel('Month')
plt.ylabel('Z-score')
plt.show()
plt.boxplot(z_score['Z-score'])
plt.show()




# Standardization of the data
y=z_score.copy()
result = adfuller(y)
adf_statistic = result[0]
p_value = result[1]

print("ADF Statistic:", adf_statistic)
print("p-value:",float(p_value) )
print('----------------------------------------------------------------------')


# # # Perform the ADF test to find d value for ARIMA
for i in range(1, 4):
    y['Z-score'] = y['Z-score'] - z_score['Z-score'].shift(i).fillna(0)
    result = adfuller(y['Z-score'])
    adf_statistic = result[0]
    p_value = result[1]

    print("ADF Statistic:", adf_statistic)
    print("p-value:", ("%.17f" % p_value).rstrip('0').rstrip('.'))


#Plotting actual and differanced values of z-score
y['Z-score'] = z_score['Z-score'] - z_score['Z-score'].shift(1).fillna(0)
plt.plot(z_score.index, z_score['Z-score'],label='Original')
plt.plot(z_score.index, y['Z-score'], label='1st differance')
plt.title('Differenced VS Original series')
plt.xlabel('Month')
plt.ylabel('Z-score')
plt.show()



#

# Plot the autocorrelation function (ACF)
plot_acf(y['Z-score'])
plt.show()

# Plot the partial autocorrelation function (PACF)
plot_pacf(y['Z-score'])
plt.show()

#Based on the analyses, we can see that our best model is ARIMA(2,1,2)



#Splitting data into train and test split
test_size=4
train_data = z_score[:(len(z_score))-test_size]
test_data = z_score[(len(z_score))-test_size:]

#Finding best orders with the help of PMDarima
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")
stepwise_fit= auto_arima(train_data, trace=True,suppress_warnings=True)
print('This is the summary for the first model',stepwise_fit.summary())

model = ARIMA(train_data, order=(3,1,1))
model_fit = model.fit()


#Forecasts for the 1st model
forecasts=model_fit.forecast(steps=len(test_data))
print('forecast',forecasts)
print('test data',test_data)
forecasts.index=test_data.index

#Mse for the 1st model

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(forecasts,test_data)
print('-----------------------Here is mse for pred vs original data',mse)



#2nd model
model2 = ARIMA(train_data, order=(2,1,1))
model_fit2 = model2.fit()
print('here is the summary for the second model',model_fit2.summary())

#Forecasts for the 2-nd model
forecasts2=model_fit2.forecast(steps=len(test_data))
print('forecast for second model',forecasts2)
print('test data',test_data)
forecasts.index=test_data.index

#Mse for the 1st model

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(forecasts2,test_data)
print('-----------------------Here is mse for pred vs original data for second model',mse)


import matplotlib.dates as mdates
forecasts2.index=test_data.index
#Plotting test data and comparing with forecasts
fig, ax = plt.subplots()
plt.plot(forecasts, label=forecasts)
plt.plot(forecasts2)
ax.plot(test_data['Z-score'], label=test_data['Z-score'])
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
#plt.title('Comparison chart of the test data and predicted values')
plt.xlabel('Date')
plt.ylabel('Z-score indicator')
location = 0 # For the best location
legend_drawn_flag = True
plt.legend(["ARIMA(3,1,1)", "ARIMA(2,1,1)",'Test Data'], loc=0, frameon=legend_drawn_flag)
from datetime import datetime
plt.show()

import matplotlib.dates as mdates
fig, ax = plt.subplots(figsize=(8, 6))

half_year_locator = mdates.MonthLocator(interval=6)
ax.xaxis.set_major_locator(half_year_locator) # Locator for major axis only.
year_month_formatter = mdates.DateFormatter("%Y-%m") # four digits for year, two for month

ax.xaxis.set_major_locator(half_year_locator)
ax.xaxis.set_major_formatter(year_month_formatter) # formatter for major axis only




# Rotates and right aligns the x labels.
# Also moves the bottom of the axes up to make room for them.
fig.autofmt_xdate()

# Rotates and right aligns the x labels.
# Also moves the bottom of the axes up to make room for them.
fig.autofmt_xdate()




#Retraining model for the full set with the following orders
model = ARIMA(z_score, order=(3,1,1))
model2 = ARIMA(z_score, order=(2,1,1))
model_fit = model.fit()
print(model_fit.summary())
model2_fit=model2.fit()
print(model2_fit.summary())
forecasts_full=model_fit.forecast(steps=5)
forecasts2_full=model2_fit.forecast(steps=5)
forecasts_full=pd.concat([z_score.iloc[-1,:],forecasts_full])
forecasts2_full=pd.concat([z_score.iloc[-1,:],forecasts2_full])
forecasts_full.index=pd.date_range(date(2023,3,1),date(2023,9,1),freq='m')
forecasts2_full.index=pd.date_range(date(2023,3,1),date(2023,9,1),freq='m')

forecasts_full=pd.DataFrame(forecasts_full, columns=['Z-score'])
forecasts2_full=pd.DataFrame(forecasts2_full, columns=['Z-score'])


plt.plot(z_score)
plt.plot(forecasts_full);
plt.plot(forecasts2_full)
plt.legend(["Original series", "Prediction (ARIMA 3,1,1)",'Prediction (ARIMA 2,1,1)'], loc=0, frameon=legend_drawn_flag)


plt.show()


print(forecasts_full)
print(forecasts2_full)
#
# 2nd model
# aic=393
# BIC=407, 270
# HQIC=398,804
#
# ll=-191,5
#
# 1st model
# ll=-190,6
# AIC=391
# BIC=405,6
# HQIQ=397/175



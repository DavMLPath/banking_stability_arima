
#to do
#fix the code to automate resuults
#solve stationarity problem fully and compare rmse
#reconstruct model predictions after detranding
# find out model waights



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

#Loading factors
data = pd.read_excel('/home/davo/Downloads/Factors.xlsx', sheet_name="Factors")
GDP_data=pd.read_excel('/home/davo/Downloads/Factors.xlsx', sheet_name="GDP")
GDP_data=GDP_data.drop(columns=[ "Unnamed: 2" ,"մլն ՀՀ դրամ "])

#Preprocessing factors
#   renaming month column in data table to match it to another . Initially it was named as ''
data.rename(columns = {' ':'Month'}, inplace = True)
#   Merging GDP and other factors
Factors = pd.merge(data, GDP_data , on='Month', how='left')
print(Factors.head())
#loading dependant varaiable


Factors['Month'] = pd.to_datetime(Factors['Month'])
z_score = pd.read_excel('/home/davo/Downloads/Factors.xlsx', sheet_name="Z-score")
#print(Factors['Month'].head())
Factors.set_index('Month', inplace=True)
print(z_score.head())
print(z_score['Month'])
plt.plot(z_score['Month'], z_score['Z-score'])
plt.xlabel('Month')
plt.ylabel('Z-score')
plt.show()
plt.boxplot(z_score['Z-score'])
plt.show()

z_score.set_index('Month', inplace=True)
z_score.drop(['Unnamed: 2', 'Unnamed: 3'],axis=1, inplace=True)
print(z_score.columns)


#Standardization of the data
from statsmodels.tsa.stattools import adfuller
# Create a sample time series
data = pd.Series(z_score['Z-score'])
from scipy.signal import detrend
# Load the time series data
# Detrend the data

detrended_data = detrend(data.values)
detrended_data = pd.DataFrame(detrended_data, index=data.index).dropna()
#detrended_data = pd.Series(detrended_data)
#detrended_data=np.log(detrended_data)
# Convert the detrended data back to a pandas dataframe

# print(detrended_data.isna())
# Plot the detrended data
#log_data = np.log(detrended_data)
#print(pd.DataFrame(log_data, index=data.index).head())
# Perform the ADF test
result = adfuller(data)
# Print the p-value and test statistic
print('ADF Statistic:', result[0])
print('p-value:', result[1])
# Perform the ADF test
result_det = adfuller(detrended_data)
# Print the p-value and test sta
print('ADF Statistic:', result_det[0])
print('p-value:', result_det[1])
# # Perform the ADF test
# result_det = adfuller(log_data)
# # Print the p-value and test sta
# print('ADF Statistic:', log_data[0])

z_score['Z-score']=detrended_data
train_data = z_score[:int(0.8*(len(z_score)))]
test_data = z_score[int(0.8*(len(z_score))):]
print('this is the test data', test_data)
# Create an instance of the ARIMA model
print('this is the number of NA-s',z_score.isna().count())
print(z_score.head())
# Create an instance of the ARIMA model


order_=[]
rmse_=[]
for i in range(1,13):
    model = ARIMA(train_data, order=(i,  1, 1))
    # Fit the model
    model_fit = model.fit()
    #print(model_fit.summary())
    # Make predictions on the test data
    predictions = model_fit.forecast(steps=len(test_data))
    print("Those are the estimated values for the test data", predictions)
    # Calculate MSE and RMSE
    mse = np.mean((predictions - test_data['Z-score'].values)**2)
    rmse = np.sqrt(mse)
    #print('mse is ', mse)
    #print('rmse is ', rmse)
    order_.append(i)
    rmse_.append(rmse)
x_axis = order_
y_axis = rmse_

plt.plot(x_axis, y_axis)
plt.title('title name')
plt.xlabel('Order')
plt.ylabel('rmse')
plt.show()

print(rmse_)


optimal_model=ARIMA(train_data, order=(12,  1, 1))
model_fit = optimal_model.fit()
print(model_fit.summary())


opt_predictions = model_fit.forecast(steps=len(test_data))

opt_mse = np.mean((opt_predictions - test_data['Z-score'].values) ** 2)
opt_rmse = np.sqrt(opt_mse)
print('this is rmse for 12.1.1 order', opt_rmse)



#
#
# test_data['pred_vales']=opt_predictions
# plt.plot(test_data)
# plt.title('title name')
# plt.xlabel('Order')
# plt.ylabel('rmse')
# plt.show()




from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")
stepwise_fit= auto_arima(train_data, trace=True,suppress_warnings=True)
stepwise_fit.summary()




# my next srteps are:::
1) make series stationary
2) decide order based on the auto_arima for full sample
3) Try that order on train data and compare test data
4) after good results, predict future 10 years
5) detrend predictions to get actual results

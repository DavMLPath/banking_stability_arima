import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")


z_score = pd.read_excel('/home/davo/Downloads/Factors.xlsx', sheet_name="Z-score")
plt.plot(z_score['Month'], z_score['Z-score'])
plt.title('Original chart')
plt.xlabel('Month')
plt.ylabel('Z-score')
#plt.boxplot(z_score['Z-score'])
f = plt.figure()
f.set_figwidth(16)
f.set_figheight(4)
plt.show()


z_score.set_index('Month', inplace=True)
z_score.drop(['Unnamed: 2', 'Unnamed: 3'],axis=1, inplace=True)
print(z_score.columns)

z_score_ori=z_score.copy()

#Standardization of the data
from statsmodels.tsa.stattools import adfuller
# Create a sample time series
from scipy.signal import detrend


z_score['Z-score']=z_score['Z-score']-z_score['Z-score'].shift(3)


plt.plot(z_score.index, z_score['Z-score'])
plt.title('3 month differenced')
plt.xlabel('Month')
plt.ylabel('Z-score')
f = plt.figure()
f.set_figwidth(4)
f.set_figheight(4)
plt.show()

data=pd.Series( z_score['Z-score']).dropna()

# Perform the ADF test
result = adfuller(data)
# Print the p-value and test statistic
print('ADF Statistic:', result[0])
print('p-value:', result[1])
# Perform the ADF test
# result_det = adfuller(detrended_data)
# # Print the p-value and test sta
# print('ADF Statistic:', result_det[0])
# print('p-value:', result_det[1])


#
# #Splitting the data into test and train sets
#
# z_score['Z-score']=detrended_data
train_data = z_score[:int(0.8*(len(z_score)))]
test_data = z_score[int(0.8*(len(z_score))):]
#print('this is the test data', test_data)
# # Create an instance of the ARIMA model
# print('this is the number of NA-s',z_score.isna().count())
# print(z_score.head())
# # Create an instance of the ARIMA model
#
#
#
#running arima for different orders

order_=[]
rmse_=[]
for i in range(1,13):
    model = ARIMA(train_data, order=(i,  1, 1))
    # Fit the model
    model_fit = model.fit()
    print(model_fit.summary())
    # Make predictions on the test data
    predictions = model_fit.forecast(steps=len(test_data))
    print("Those are the estimated values for the test data", predictions)
    # Calculate MSE and RMSE
    mse = np.mean((predictions - test_data['Z-score'].values)**2)
    rmse = np.sqrt(mse)
    print('mse is ', mse)
    print('rmse is ', rmse)
    order_.append(i)
    rmse_.append(rmse)
x_axis = order_
y_axis = rmse_

#plotting arima for different orders
plt.plot(x_axis, y_axis)
plt.title('title name')
plt.xlabel('Order')
plt.ylabel('rmse')
plt.show()



# #using optimal i value
data=train_data
optimal_model=ARIMA(data, order=(12,  1, 1))
model_fit = optimal_model.fit()
print(model_fit.summary())
#
#
opt_predictions = model_fit.forecast(steps=len(test_data))
opt_mse = np.mean((opt_predictions - test_data['Z-score'].values) ** 2)
opt_rmse = np.sqrt(opt_mse)
print('this is rmse for 12.1.1 order', opt_rmse)
#
#
#compering predicrtions with optimal values
#
#print(test_data.head())
#print('the type of predictions is' , type(opt_predictions))
#print('the type of test data is' , type(test_data))
test_data=pd.DataFrame(test_data)
test_data['pred_vales']=opt_predictions
plt.plot(test_data)
plt.plot(test_data)
plt.title('title name')
plt.xlabel('Order')
plt.ylabel('rmse')
plt.show()




#rearranging results, and training model on the full dataset






plt.plot(z_score)
#plt.plot(z_score_ori)
plt.show()


# step 1use shifted results== data , to train another model on the same order


data=z_score
optimal_model=ARIMA(data, order=(12,  1, 1))
model_fit = optimal_model.fit()
print(model_fit.summary())
#
#
opt_predictions = model_fit.forecast(steps=10)




plt.plot(opt_predictions)
plt.show()
#
# print(z_score_ori.tail(10))
# print(z_score.tail(10))
# print(opt_predictions.index)
# print(type(opt_predictions))
opt_predictions=opt_predictions.to_frame('Z-score')
# print(opt_predictions.head())
#
total_series=pd.concat([z_score_ori,opt_predictions])



l=list(total_series['Z-score'])
l_n=[]
for  i in range(len(l)):
    if i>=124:
        l[i]+=l[i-3]
        l_n.append(l[i])
    print(l[i])


print('this is the index', total_series.index)

recovered=pd.DataFrame(l_n)
print('here are the recovered series',recovered)
print('here is the type', type(total_series))
print('check',total_series.iloc[124:126])
print(z_score.head(10))
#

# plt.plot(l_n)
# plt.plot(total_series)
# plt.show()
#
#
# print(len(l_n),len(l))

#
# t1=z_score.tail(20).shift(3)
# t2=z_score.tail(20)
#
# print('this is t1',t1)
# print('this is t2',t2)




#print(l_n)
# print(len(z_score_ori))
# print(len(opt_predictions))
# #
# print(len(total_series))
#
#
#

#now lets try to use our model and train on entire data, after which we can predict next 10 years


#now let's recover predictions  from their shifted values, so we need to add values from previews 3 months to the actual values






#
#
#
# #finding best model with auto_arima library
#
#
# from pmdarima import auto_arima
# import warnings
# warnings.filterwarnings("ignore")
# stepwise_fit= auto_arima(train_data, start_p=0,  start_q=0,  max_p=4, max_q=4, trace=True,suppress_warnings=True)
# stepwise_fit.summary()
#

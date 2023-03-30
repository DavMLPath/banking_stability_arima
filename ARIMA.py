import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

z_score = pd.read_excel('/home/davo/Downloads/Factors.xlsx', sheet_name="Z-score")
plt.plot(z_score['Month'], z_score['Z-score'])
plt.title('Original chart')
plt.xlabel('Month')
plt.ylabel('Z-score')
plt.boxplot(z_score['Z-score'])
f = plt.figure()
f.set_figwidth(16)
f.set_figheight(4)
#plt.show()


z_score.set_index('Month', inplace=True)
z_score.drop(['Unnamed: 2', 'Unnamed: 3'],axis=1, inplace=True)
print(z_score.columns)


plt.plot(z_score.index, z_score['Z-score'])
plt.title('Z-score trend over time')
plt.xlabel('Month')
plt.ylabel('Z-score')
f = plt.figure()
f.set_figwidth(4)
f.set_figheight(4)
#plt.show()

z_score_ori=z_score.copy()


#Standardization of the data
#Differencing for a 3 months
z_score['Z-score']=z_score['Z-score']-z_score['Z-score'].shift(3)


plt.plot(z_score.index, z_score['Z-score'])
plt.title('3 month differenced')
plt.xlabel('Month')
plt.ylabel('Z-score')
f = plt.figure()
f.set_figwidth(4)
f.set_figheight(4)
#plt.show()

data=pd.Series( z_score['Z-score']).dropna()

# Perform the ADF test
result = adfuller(data)
# Print the p-value and test statistic
print('ADF Statistic:', result[0])
print('p-value:', result[1])


#Splitting data into train and test split
train_data = z_score[:int(0.8*(len(z_score)))]
test_data = z_score[int(0.8*(len(z_score))):]
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
plt.title('RMSE change for different orders of ARIMA')
plt.xlabel('Order')
plt.ylabel('rmse')
#plt.show()



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



#compering predicrtions with optimal values
test_data=pd.DataFrame(test_data)
test_data['pred_vales']=opt_predictions
test_data.plot()
plt.title('Compeaision of predictions with original')
plt.xlabel('Date')
plt.ylabel('Z-score')
plt.show()


print(test_data.head(10))
print(test_data.tail(10))

# #rearranging results, and training model on the full dataset
#
# step 1use shifted results== data , to train another model on the same order


data=z_score
optimal_model=ARIMA(data, order=(12,  1, 1))
model_fit = optimal_model.fit()
print(model_fit.summary())
#
#
opt_predictions = model_fit.forecast(steps=10)



plt.plot(opt_predictions)
plt.title('Predected values')
plt.show()
opt_predictions=opt_predictions.to_frame('Z-score')

"""
Now lets concat predictions with original data, 
for recovering shifted values
"""


total_series=pd.concat([z_score_ori,opt_predictions])

l=list(total_series['Z-score'])
l_n=[]

for  i in range(len(l)):
    if i>=124:
        total_series.iloc[i,0]+=total_series.iloc[i-3,0]

print(total_series.head(10))
print(total_series.tail(10))
print(len(total_series))






#After this, we can plot original series and predections, in upcoming years



ax = total_series.iloc[:124,:].plot(ls="-", color="b")
ax2 = ax.twinx()           #Create a twin Axes sharing the xaxis

total_series.iloc[124:,:].plot(ls="--", color="r", ax=ax)
plt.axhline(y=0.5,linestyle="--",animated=True,label="False Alaram")

plt.show()


# After this step, we can now download a new series,
# where we have original values for 2022, and see weather it makes scense or no

#Then I can open a new branch in git hub and push my code, for assepting changes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

z_score = pd.read_excel('C:/Users/37493/Downloads/Factors.xlsx', sheet_name="Z-score")


z_score.set_index('Month', inplace=True)
z_score.drop(['Unnamed: 2', 'Unnamed: 3'],axis=1, inplace=True)
print(z_score.head())

train_data = z_score[:int(0.8*(len(z_score)))]
test_data = z_score[int(0.8*(len(z_score))):]
print(train_data.tail())

plt.plot(train_data)
plt.xlabel('Year')
plt.ylabel('Data')
plt.show()



# Plot the autocorrelation function (ACF)
plot_acf(train_data)
plt.show()

# Plot the partial autocorrelation function (PACF)
plot_pacf(train_data)
plt.show()

#
# model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
# results = model.fit()
# predictions = results.forecast(steps=len(test_data))
#
# print(train_data.tail())
#
# print(predictions)
#
#
#


# Define the range of p, d, and q values
p_values = range(0, 3)
d_values = range(0, 2)
q_values = range(0, 3)

# Loop through all possible combinations of p, d, and q
best_model = None
best_rmse = np.inf
q_={}
for p in p_values:
    for d in d_values:
        for q in q_values:
                model = SARIMAX(train_data, order=(p, d, q), seasonal_order=(1, 1, 1, 12))
                results = model.fit()
                predictions = results.forecast(steps=len(test_data))
                mse = mean_squared_error(test_data, predictions)
                rmse = np.sqrt(mse)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
                q_[rmse]=[p,d,q]

#
print('initial best rmse is',best_rmse)
best_result=best_model.fit()
best_predictions = best_result.forecast(steps=len(test_data))
best_mse = mean_squared_error(test_data, best_predictions)

print('final best rmse is',best_rmse)
print(best_predictions)


# by usinf seasonal order as 12, model gives RMSE 1.64
plt.plot(best_predictions, c='r')
plt.plot(test_data, c='b')
plt.show()


for i in q_.keys():
    if i==best_rmse:
        print(q_[i], 'whoalla')




final_model = SARIMAX(z_score, order=(2, 1, 1), seasonal_order=(1, 1, 1, 12))
final_result=final_model.fit()
final_predictions = final_result.forecast(steps=10)

print(final_predictions)
plt.plot(final_predictions)
plt.show()

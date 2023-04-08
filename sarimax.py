import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
​
z_score = pd.read_excel('/home/davo/Downloads/Factors.xlsx', sheet_name="Z-score")
​
z_score.set_index('Month', inplace=True)
z_score.drop(['Unnamed: 2', 'Unnamed: 3'],axis=1, inplace=True)
print(z_score.head())
​
z_score.plot()
plt.show()
import statsmodels.api as sm
​
model=sm.tsa.statespace.SARIMAX(z_score,order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()
z_score['forecast']=results.predict(start=len(z_score)-10,end=len(z_score))
print(z_score.columns)
z_score[['Z-score','forecast']].plot(figsize=(12,8))
plt.show()
​
​
z_score['forecast']=results.predict(steps=10)
print(z_score.columns)
z_score['forecast'].plot()
plt.show()
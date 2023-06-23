#!/usr/bin/env python
# coding: utf-8

# In[537]:


import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pmdarima as pm
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import acf, pacf
from datetime import datetime
from scipy import stats
from scipy.stats import boxcox
import statsmodels.tsa as smt


import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[538]:


df = pd.read_csv('final_data.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.set_index('DateTime', inplace = True)
df


# ## PM 2.5

# In[539]:


df['PM2.5'].plot()


# In[540]:


df.shift(1)


# In[541]:


df.iloc[:,1:].reset_index(drop = True)


# In[542]:


plt.scatter(df.shift(1)['PM2.5'].iloc[1:],df['PM2.5'].iloc[1:])
plt.xlabel('Previous Day PM 2.5')
plt.ylabel('PM 2.5')


# In[543]:


decomposition= sm.tsa.seasonal_decompose(df['PM2.5'], model='additive')

ax = decomposition.plot()
ax.set_size_inches(10,10)

plt.show()


# ## O3

# In[544]:


df['O3'].plot()


# In[545]:


plt.scatter(df.shift(1)['O3'].iloc[1:],df['O3'].iloc[1:])
plt.xlabel('Previous Day O3')
plt.ylabel('O3');


# In[546]:


decomposition= sm.tsa.seasonal_decompose(df['O3'], model='additive')

ax = decomposition.plot()
ax.set_size_inches(10,10)

plt.show()


# In[547]:


df


# In[548]:


# ADF test

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import acf, pacf
from datetime import datetime
from scipy import stats
from scipy.stats import boxcox
import statsmodels.tsa as smt

adf_test = adfuller(df['O3'].diff(periods=1).iloc[1:])

print('ADF Statistic: %f' % adf_test[0])
print('Critical Values @ 0.05: %.2f' % adf_test[4]['5%'])
print('p-value: %20.20f' % adf_test[1])


# ## Multivariate LSTM

# In[549]:


# Splitting for train and test data - 80% trainig set and 20% test set


# In[2248]:


df


# In[2249]:


df.iloc[:877,1:]


# In[2466]:


df_train = df.iloc[:877,1:].reset_index(drop = True)
df_test = df.iloc[877:,1:].reset_index(drop = True)


# In[2467]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df_train_scaled = sc.fit_transform(df_train)
print(df_train_scaled.shape)


# In[2468]:


sc2 = StandardScaler()
y_train = sc2.fit_transform(df_train[['O3']])
y_train.shape


# In[2469]:


window_size = 5
X_train = []
y_train1 = []
for i in range(5,len(df_train_scaled)-4):
    X_train.append(df_train_scaled[i-5:i])
    y_train1.append(y_train.flatten()[i:(i+5)])
X_train, y_train1 = np.array(X_train), np.array(y_train1)


# In[2470]:


y_train1


# In[2471]:


from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization


# In[2472]:


from tensorflow.keras import layers, callbacks

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
    
)


# In[2473]:


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid= train_test_split(X_train,y_train1, test_size=0.1, random_state=0)


# In[2474]:


tf.random.set_seed(42)
model = Sequential()
model.add(LSTM(128,  activation = 'linear',return_sequences=True, kernel_regularizer=regularizers.L2(0.01),input_shape=(5,3)))
#model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(LSTM(128,activation = 'linear', kernel_regularizer=regularizers.L2(0.01)))
#model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(5, kernel_regularizer=regularizers.L2(0.01)))
model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))


history = model.fit(X_train, y_train,
    batch_size=128,
    epochs=200,validation_data=(X_valid, y_valid),callbacks = [early_stopping])


# In[2475]:


# plot train and validation loss
from matplotlib import pyplot
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()


# ## Preparation of test dataset

# In[2476]:


df_train_last7 = df_train.iloc[-5:]
df_test1 = pd.concat((df_train_last7, df_test), axis = 0)
df_test1.head()


# In[2477]:


df_test2 = sc.transform(df_test1)


# In[2478]:


y_t = sc2.transform(df_test1[['O3']])


# In[2479]:


window_size = 5
X_test = []
y_test1 = []
for i in range(5,len(df_test1)-4):
    X_test.append(df_test2[i-5:i])
    y_test1.append(y_t.flatten()[i:(i+5)])
X_test, y_test1 = np.array(X_test), np.array(y_test1)


# In[2480]:


y_test = model.predict(X_test)


# In[2481]:


evaluation = pd.DataFrame({'Metric': ['MSE'],
                           'Training set': [mean_squared_error(y_train, model.predict(X_train))],
                           'Testing set': [mean_squared_error(y_test1.flatten(), y_test.flatten())]})
evaluation = evaluation.set_index('Metric')
evaluation


# In[2482]:


y_test = sc2.inverse_transform(y_test)
y_test


# In[2483]:


final_df = pd.DataFrame(y_test.flatten())
final_df.columns = ['y_test']
final_df


# In[2484]:


y_test1 = sc2.inverse_transform(y_test1)
final_df['true value'] = y_test1.flatten()


# In[2485]:


final_df


# In[2486]:


plt.plot(final_df['y_test'].iloc[-800:], label = 'predicted', color = 'red')
plt.plot(final_df['true value'].iloc[-800:], label = 'true')
pyplot.legend(['predicted', 'true'], loc='upper right')
plt.show()


# In[2487]:


from sklearn.metrics import mean_squared_error, r2_score
mean_squared_error(y_test1.flatten(), y_test.flatten())


# In[2488]:


import datetime
start = datetime.datetime.strptime("2022-01-01", "%Y-%m-%d")


# In[2489]:


from dateutil.relativedelta import relativedelta
date_list = [start + relativedelta(days=x) for x in range(0,5)]
future_prediction = pd.DataFrame(index=date_list, columns= df.columns)
df1= pd.concat([df, future_prediction])
df1


# In[2490]:


X_test1 = []
X_test1.append(df1.iloc[-10:-5,1:])
X_test1 = np.array(X_test1)
X_test1 = sc.transform(X_test1.reshape(-1, X_test1.shape[-1])).reshape(X_test1.shape)


# In[2491]:


y = sc2.inverse_transform(model.predict(X_test1))
y


# In[2492]:


l = []
l.append(df1.iloc[-10:-5,1:].values[-1][0])
for i in range(len(y[-1])):
    l.append(y[-1][i])
l


# In[2493]:


lastData = df1.iloc[-35:-5,1].values


# In[2494]:


ax1 = np.arange(1, len(lastData) + 1)

ax2 = np.arange(len(lastData), len(lastData) + len(l))

plt.figure(figsize=(8, 4))

plt.plot(ax1, lastData, 'b-o', color='blue', markersize=3, label='Time series', linewidth=1)

plt.plot(ax2, l, 'b-o', color='red', markersize=3, label='Estimate')

plt.axvline(x=ax1[-1],  linestyle='dashed', linewidth=1)

plt.legend()
plt.ylim(0, 0.08)
plt.axhline(y =0.06,  linestyle='dashed', linewidth=1, color = 'red')
plt.show()


# ## Univariate LSTM for O3

# In[2111]:


df_train = df.iloc[:877,:].reset_index(drop = True)
df_test = df.iloc[877:,:].reset_index(drop = True)


# In[1385]:


from tensorflow.keras import regularizers


# In[1530]:


tf.random.set_seed(1234)
print(tf.random.uniform([1]).numpy())
print(tf.random.uniform([1]).numpy())


# In[2112]:


from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(2)
scaler = StandardScaler()
y_train1 = scaler.fit_transform(df_train[['O3']])
# generate the input and output sequences
n_lookback = 60 # length of input sequences (lookback period)
n_forecast = 30 # length of output sequences (forecast period)



X = []
Y = []
for i in range(n_lookback, len(y_train1) - n_forecast + 1):
    X.append(y_train1.flatten()[i - n_lookback: i])
    Y.append(y_train1.flatten()[i: i + n_forecast])

X = np.array(X)
Y = np.array(Y)
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid= train_test_split(X,Y, test_size=0.1, random_state=42)

# fit the model
model = Sequential()
model.add(LSTM(128,  activation = 'linear', return_sequences=True, input_shape=(60,1)))
model.add(Dropout(0.3))
#model.add(BatchNormalization())
model.add(LSTM(128,  activation = 'linear'))
model.add(Dropout(0.3))
#model.add(BatchNormalization())
model.add(Dense(30))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss='mse')
history = model.fit(X_train, y_train,
    batch_size=128,
    epochs=200,validation_data=(X_valid, y_valid))


#kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.01)


# In[2113]:


# plot train and validation loss
from matplotlib import pyplot
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()


# In[2118]:


df_train_last7 = df_train.iloc[-60:]
df_test1 = pd.concat((df_train_last7, df_test), axis = 0)
df_test1.tail()


# In[2119]:


y_test2 = scaler.transform(df_test1[['O3']])


# In[2120]:


X_test = []
y_test = []

for i in range(n_lookback, len(y_test2) - n_forecast + 1):
    X_test.append(y_test2.flatten()[i - n_lookback: i])
    y_test.append(y_test2.flatten()[i: i + n_forecast])


# In[2121]:


X_test = np.array(X_test)
y_test = np.array(y_test)


# In[2122]:


y_test


# In[2123]:


y_pred = model.predict(X_test)


# In[2124]:


from sklearn.metrics import mean_squared_error
evaluation = pd.DataFrame({'Metric': ['MSE'],
                           'Training set': [mean_squared_error(y_train.flatten(), model.predict(X_train).flatten())],
                           'Testing set': [mean_squared_error(y_test.flatten(), model.predict(X_test).flatten())]})
evaluation = evaluation.set_index('Metric')
evaluation


# In[2125]:


y_pred = scaler.inverse_transform(y_pred)
final_df = pd.DataFrame(y_pred.flatten())
final_df.columns = ['y_test']
final_df


# In[2126]:


y_test1 = scaler.inverse_transform(y_test)
final_df['true value'] = y_test1.flatten()


# In[2127]:


final_df


# In[2128]:


plt.plot(final_df['y_test'], label = 'predicted', color = 'red')
plt.plot(final_df['true value'], label = 'true')
pyplot.legend(['predicted', 'true'], loc='upper right')
plt.show()


# In[2129]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test1.flatten(), y_pred.flatten())


# In[2130]:


y_test


# In[2131]:


# generate the forecasts
X_ = y_test2[- n_lookback:]  # last available input sequence
X_ = X_.reshape(1, n_lookback, 1)

Y_ = model.predict(X_).reshape(-1, 1)
Y_ = scaler.inverse_transform(Y_)

# organize the results in a data frame
df_past = df[['O3']].reset_index()
df_past.rename(columns={'DateTime': 'Date', 'O3': 'Actual'}, inplace=True)
df_past['Date'] = pd.to_datetime(df_past['Date'])
df_past['Forecast'] = np.nan
df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
df_future['Forecast'] = Y_.flatten()
df_future['Actual'] = np.nan

results = df_past.append(df_future).set_index('Date')

# plot the results
results.iloc[-400:].plot(title='O3')


# In[ ]:


## Setting the dynamically changing learnin rate based on epochs number
# Set the learning rate scheduler
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))

history = model_tune.fit(dataset, epochs=100, callbacks=[lr_schedule])


# Define the learning rate array
lrs = 1e-8 * (10 ** (np.arange(100) / 20))

# Set the figure size
plt.figure(figsize=(10, 6))

# Set the grid
plt.grid(True)

# Plot the loss in log scale
plt.semilogx(lrs, history.history["loss"])

# Increase the tickmarks size
plt.tick_params('both', length=10, width=1, which='both')

# Set the plot boundaries
plt.axis([1e-8, 1e-3, 0, 300])

#You will then set the optimizer with a learning rate close to the minimum. It is set to 4e-6 initially but feel free to change based on your results.


# In[1310]:


x = [0,11,24,37,49,59]
print(x)
z = pm.utils.diff(x,lag=1,differences=1)
print(z)
z = np.insert(z,0,x[0])
print(z)
print(np.cumsum(z))


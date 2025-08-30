# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from itertools import product
from math import sqrt
from sklearn.metrics import mean_squared_error
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set visualization settings
sns.set(rc={"figure.figsize": (20, 10), "axes.titlesize": 18, "axes.labelsize": 12,
            "xtick.labelsize": 14, "ytick.labelsize": 14})

# Load and preprocess data
dateparse = lambda dates: datetime.strptime(dates, '%Y-%m-%d')
df = pd.read_csv('../input2/all_currencies.csv', parse_dates=['date'], index_col='date', date_parser=dateparse)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.sample(5)

# Extract Bitcoin data and drop unnecessary columns
btc = df[df['symbol'] == 'BTC']
btc.drop(['volume', 'market'], axis=1, inplace=True)

# Resample to monthly frequency
btc_month = btc[['close']].resample('M').mean()


# Stationarity check and seasonal decomposition
seasonal_decompose(btc_month.close, model='additive').plot()
print("Dickey–Fuller test: p=%f" % adfuller(btc_month.close)[1])

# Box-Cox Transformation
btc_month['close_box'], lmbda = stats.boxcox(btc_month.close)
print("Dickey–Fuller test: p=%f" % adfuller(btc_month.close_box)[1])

# Seasonal Differentiation (12 months)
btc_month['box_diff_seasonal_12'] = btc_month.close_box - btc_month.close_box.shift(12)
print("Dickey–Fuller test: p=%f" % adfuller(btc_month.box_diff_seasonal_12[12:])[1])

# Seasonal Differentiation (3 months)
btc_month['box_diff_seasonal_3'] = btc_month.close_box - btc_month.close_box.shift(3)
print("Dickey–Fuller test: p=%f" % adfuller(btc_month.box_diff_seasonal_3[3:])[1])

# Regular Differentiation
btc_month['box_diff2'] = btc_month.box_diff_seasonal_12 - btc_month.box_diff_seasonal_12.shift(1)
seasonal_decompose(btc_month.box_diff2[13:]).plot()
print("Dickey–Fuller test: p=%f" % adfuller(btc_month.box_diff2[13:])[1])

# Autocorrelation and Partial Autocorrelation
ax = plt.subplot(211)
plot_acf(btc_month.box_diff2[13:].values.squeeze(), lags=12, ax=ax)
ax = plt.subplot(212)
plot_pacf(btc_month.box_diff2[13:].values.squeeze(), lags=12, ax=ax)
plt.tight_layout()

# ARIMA Parameter Selection
qs = range(0, 3)
ps = range(0, 3)
d = 1
parameters = product(ps, qs)
parameters_list = list(parameters)

results = []
best_aic = float("inf")
for param in parameters_list:
    try:
        model = SARIMAX(btc_month.close_box, order=(param[0], d, param[1])).fit(disp=-1)
    except ValueError:
        print('bad parameter combination:', param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])

# Display best model
result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print(result_table.sort_values(by='aic', ascending=True).head())
print(best_model.summary())

# Model Diagnostics
best_model.plot_diagnostics(figsize=(15, 12))
plt.show()

# Inverse Box-Cox Transformation Function
def invboxcox(y, lmbda):
    if lmbda == 0:
        return np.exp(y)
    else:
        return np.exp(np.log(lmbda * y + 1) / lmbda)

# Prediction
btc_month_pred = btc_month[['close']]
date_list = [datetime(2018, 6, 30), datetime(2018, 7, 31), datetime(2018, 8, 31)]
future = pd.DataFrame(index=date_list, columns=btc_month.columns)
btc_month_pred = pd.concat([btc_month_pred, future])

btc_month_pred['forecast'] = invboxcox(best_model.predict(start=datetime(2014, 1, 31), end=datetime(2018, 8, 31)), lmbda)

btc_month_pred.close.plot(linewidth=3)
btc_month_pred.forecast.plot(color='r', ls='--', label='Predicted Close', linewidth=3)
plt.legend()
plt.grid()
plt.title('Bitcoin monthly forecast')
plt.ylabel('USD')
plt.show()
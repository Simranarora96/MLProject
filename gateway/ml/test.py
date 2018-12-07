from pandas import Series
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from pandas import DataFrame
from pandas import read_csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
import numpy as np
from sklearn.metrics import accuracy_score

dataframe = DataFrame()
dataframe_test = DataFrame()
frame_train = DataFrame()
frame_test = DataFrame()
dataframe = read_csv('lags_12months_featurespower_trainset.csv', header=0)

# split into input and output
features = dataframe[['t','t-6','t-4','t-2']].copy()
array_feature = features.values
X = array_feature[2:226,:]
X_test = array_feature[227:276,:]
print(X.shape)
print(X_test.shape)
frame_train = read_csv('Trainset.csv', header = 0)

array_frame_train = frame_train.values

y = array_frame_train[27:251,1]


y_test = array_frame_train[252:301,1] 

print(y.shape)
print(y_test.shape)
print('''''''''''')

model = RandomForestRegressor(n_estimators=2500, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)
errors = abs(predictions - y_test)
print(predictions)
print(predictions.shape)
# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = np.mean(100 * (errors / y_test))
print(mape)
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')
print(model.score(X_test, y_test))
# line plot of time series
from pandas import Series
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from pandas import DataFrame
from pandas import read_csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
# load dataset
series = Series.from_csv('Trainset.csv', header=0)
# display first few rows
print(series.head(5))
# line plot of dataset
series.plot()
pyplot.show()

# seasonally adjust the time series
# seasonal difference
differenced = series.diff(12)
# trim off the first year of empty data
differenced = differenced[12:]
# save differenced dataset to file
differenced.to_csv('seasonally_adjustedpower_trainset.csv')
# plot differenced dataset
differenced.plot()
pyplot.show()

series = Series.from_csv('seasonally_adjustedpower_trainset.csv', header=None)
plot_acf(series)
pyplot.show()


# reframe as supervised learning
dataframe = DataFrame()
for i in range(12,0,-1):
	dataframe['t-'+str(i)] = series.shift(i)
	dataframe['t'] = series.values
print(dataframe.head(13))
dataframe = dataframe[13:]
# save to new file
dataframe.to_csv('lags_12months_featurespower_trainset.csv', index=False)


# load data
dataframe = read_csv('lags_12months_featurespower_trainset.csv', header=0)
array = dataframe.values
# split into input and output

X = array[:,0:-1]
y = array[:,-1]
# fit random forest model
model = RandomForestRegressor(n_estimators=500, random_state=1)
model.fit(X, y)
# show importance scores
print(model.feature_importances_)
# plot importance scores
names = dataframe.columns.values[0:-1]
ticks = [i for i in range(len(names))]
pyplot.bar(ticks, model.feature_importances_)
pyplot.xticks(ticks, names)
pyplot.show()

# separate into input and output variables
array = dataframe.values
X = array[:,0:-1]
y = array[:,-1]
# perform feature selection
rfe = RFE(RandomForestRegressor(n_estimators=500, random_state=1), 4)
fit = rfe.fit(X, y)
# report selected features
print('Selected Features:')
names = dataframe.columns.values[0:-1]
for i in range(len(fit.support_)):
	if fit.support_[i]:
		print(names[i])
# plot feature rank
names = dataframe.columns.values[0:-1]
ticks = [i for i in range(len(names))]
pyplot.bar(ticks, fit.ranking_)
pyplot.xticks(ticks, names)
pyplot.show()
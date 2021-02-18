""" Ao inves de normalizar os valores por dia, calcular a variação de valores
de um candle pra outro"""

from Data_treatment.DT_1 import StockData
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

stockdata = StockData('WIN$', date=datetime.today(), window_days=10, timeframe=5)
stockdata.data_final(n_in=5)

train_x = stockdata.train_x.values
train_y = stockdata.train_y.values
x_val = stockdata.val_x.values
y_val = stockdata.val_y.values

train_x = train_x.reshape(train_x.shape[0], -1)
x_val = x_val.reshape(x_val.shape[0], -1)

clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(train_x, train_y)
# Predict the response for test dataset
y_pred = clf.predict(x_val).reshape(-1, 1)
y_val = y_val.reshape(y_val.shape[0], -1)
print("Accuracy:", metrics.accuracy_score(y_val, y_pred[:-1]))
a = 1

from Data_treatment.Data_Treatment import StockData_Bin
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn import metrics

# stockdata = DecisionTreeeData('WIN$', window_days=330, timeframe=5, date=datetime.today())

stockdata = StockData_Bin("WIN$", 5, 1)
stockdata.stockdata(5, datetime.today(), 200)

# X = data.tick_frame_train.loc[:, ['open', 'high', 'low', 'close', 'real_volume', 'RSI', 'K']]
# X_test = data.tick_frame_test.loc[:, ['open', 'high', 'low', 'close', 'real_volume', 'RSI', 'K']]
# y = data.tick_frame_train.loc[:, 'close_dif']
# y_test = data.tick_frame_test.loc[:, 'close_dif']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

train_x = stockdata.train_x
train_y = stockdata.train_y
x_val = stockdata.x_today
y_val = stockdata.y_today

train_x = train_x.reshape(train_x.shape[0], -1)
x_val = x_val.reshape(x_val.shape[0], -1)

# X, x_val, y_train, y_val = split(X, y, test_size=0.20, random_state=1120)

clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(train_x, train_y)
# Predict the response for test dataset
y_pred = clf.predict(x_val).reshape(-1, 1)
y_val = y_val.reshape(y_val.shape[0], -1)
print("Accuracy:", metrics.accuracy_score(y_val[:-1], y_pred[:-1]))

a = 1

import numpy as np
from keras.models import load_model
from DataSet.DataSet import StockData_robo
from datetime import datetime


def calcregr():
    stockdata = StockData_robo('WIN$', date=datetime.today(), timeframe=15)
    stockdata.data_final(n_in=17)
    train_x = stockdata.train_x
    train_x = train_x.reshape(*train_x.shape, 1)
    y_pred = autoencoder.predict(train_x)
    x_train = y_pred.reshape(y_pred.shape[0], -1)
    y_pred = regressao.predict(x_train)
    y_pred = np.argmax(y_pred).astype(str)
    return y_pred


regressao = load_model(r'C:\Users\mueld\Documents\Python_Projects\Stock_1\model_linear.h5')
autoencoder = load_model(r'C:\Users\mueld\Documents\Python_Projects\Stock_1\model_autoencoder.h5')

calcregr()

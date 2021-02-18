from keras import models
from Data_treatment.Data_Treatment import StockData
from MetaTrader5 import TIMEFRAME_M5
from datetime import datetime
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error

StockData = StockData("WINZ20", 10, 1)
StockData.download_stock(TIMEFRAME_M5, datetime.today(), 20)
StockData.values_norm()

PATH = r'D:/PYTHON/reinforcement-learning-using-python-master/models_s/'
r2_h = 10e20
r2_l = 10e20
for path in os.listdir(PATH):
    model = models.load_model(PATH + path)
    predict = model.predict(StockData.train_x_today)
    predicted = StockData.scaler_y.inverse_transform(predict)
    r2_h_1 = mean_absolute_error(predicted[:, 0], StockData.real_y[:, 0])
    r2_l_1 = mean_absolute_error(predicted[:, 1], StockData.real_y[:, 1])
    if r2_h_1 < r2_h:
        r2_h = r2_h_1
        print('best High = {}'.format(r2_h))
        path_best_h = path
        predict_h = predicted
        model_h = model
    if r2_l_1 < r2_l:
        r2_l = r2_l_1
        print('best_low = {}'.format(r2_l))
        predict_l = predicted
        model_l = model
        path_best_h = path

plt.title('Stock')
plt.ylabel('bvmf-pts')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(predict_h[:, 0], label='pred_max')
plt.plot(predict_l[:, 1], label='pred_min')
plt.plot(StockData.real_y[:, 0], label='real_max')
plt.plot(StockData.real_y[:, 1], label='real_min')
plt.legend()
plt.show()
a = 1

a = 1

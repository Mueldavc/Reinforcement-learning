from Data_treatment.Data_Treatment import StockData
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Flatten
from keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
from keras.optimizers import SGD
import numpy as np

# StockData = StockData("WIN$", 10, 1)
# StockData.download_stock(TIMEFRAME_M5, datetime.today(), 300)
# StockData.values_norm()
StockData = StockData("WIN$", 14, 1)
StockData.stockdata(5, datetime.today(), 330)


loss_fn = MeanSquaredError()
opt = SGD(lr=0.02, momentum=0.9)
# opt = optimizador()
train_x = StockData.train_x
print(train_x.shape)
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
test_x = StockData.x_today
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(*StockData.ENVIRONMENT_SHAPE, 1)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='sigmoid', padding='same'))

# model.add(Flatten())
# model.add(Dense(2, activation='linear'))
model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

history = model.fit(train_x, train_x, epochs=50, verbose=2, validation_split=0.2, shuffle=True, batch_size=64)
model.save(r'C:\Users\mueld\Documents\Python_Projects\Stock_1\model.h5')
predict = model.predict(test_x)
predict = StockData.scaler_y.inverse_transform(predict)
predict = np.append(predict, [np.nan, np.nan]).reshape(-1, 2)
real_y = np.append([np.nan, np.nan], StockData.train_y_today).reshape(-1, 2)
plt.title('Stock')
plt.ylabel('bvmf-pts')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(predict[:, 0], label='pred_max')
plt.plot(predict[:, 1], label='pred_min')
plt.plot(real_y[:, 0], label='real_max')
plt.plot(real_y[:, 1], label='real_min')
plt.legend()
plt.show()
a = 1

# plt.subplot(211)
# plt.title('Loss')
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# # plot mse during training
# plt.subplot(212)
# plt.title('Mean Squared Error')
# plt.plot(history.history['mean_squared_error'], label='train')
# plt.plot(history.history['val_mean_squared_error'], label='test')
# plt.legend()
# plt.show()

from Data_treatment.Data_Treatment import StockData_Bin
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.utils import to_categorical
from sklearn.metrics import mean_squared_error
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten

class Cursor:
    """
    A cross hair cursor.
    """

    def __init__(self, ax):
        self.ax = ax
        self.horizontal_line = ax.axhline(color='k', lw=0.8, ls='--')
        self.vertical_line = ax.axvline(color='k', lw=0.8, ls='--')
        # text location in axes coordinates
        self.text = ax.text(0.72, 0.9, '', transform=ax.transAxes)

    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        self.text.set_visible(visible)
        return need_redraw

    def on_mouse_move(self, event):
        if not event.inaxes:
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax.figure.canvas.draw()
        else:
            self.set_cross_hair_visible(True)
            x, y = event.xdata, event.ydata
            # update the line positions
            self.horizontal_line.set_ydata(y)
            self.vertical_line.set_xdata(x)
            self.text.set_text('x=%1.2f, y=%1.2f' % (x, y))
            self.ax.figure.canvas.draw()


# test_x = StockData.train_x_today
# test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)

# model = load_model(r'C:\Users\mueld\Documents\Python_Projects\Stock_1\model.h5')
Window_Size = 28
windows_days = 112
num_units = 10
num_units_1 = 3
epocas = 3


stockdata = StockData_Bin("WIN$", Window_Size, 1)
stockdata.stockdata(5, datetime.today(), windows_days)

train_x = stockdata.train_x
train_y = to_categorical(stockdata.train_y)

train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
test_x = stockdata.x_today
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)

model = Sequential()

if num_units_1 != 0:
    model.add(Conv2D(num_units, (3, 3), activation='relu', padding='same', input_shape=(*stockdata.ENVIRONMENT_SHAPE, 1)))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(num_units_1, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(num_units_1, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(num_units, (3, 3), activation='relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(2, (3, 3), activation='sigmoid', padding='same'))

else:
    model.add(LSTM(num_units, input_shape=stockdata.ENVIRONMENT_SHAPE))


model.add(Flatten())
model.add(Dense(2, activation='sigmoid'))

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_x, train_y, epochs=epocas, batch_size=10, shuffle=True, verbose=2)
model.save(r'C:\Users\mueld\Documents\Python_Projects\Stock_1\model.h5')

predicted = model.predict(test_x)
# predicted = stockdata.scaler_y.inverse_transform(predict)
predicted = [np.argmax(i) for i in predicted]
# predicted = np.append([np.nan, np.nan], predicted).reshape(-1, 2)
# real_y = np.append(StockData.train_y_today, [np.nan, np.nan]).reshape(-1, 2)
real_y = stockdata.y_today

rmse = np.sqrt(mean_squared_error(real_y, predicted))
print('Validation RMSE: ', rmse, '\n')

# fig, axs = plt.subplots()
# axs.set_ylim([int(np.nanmin(np.append(predicted, real_y))), int(np.nanmax(np.append(predicted, real_y)))])
# # axs.set_ylim([max(max(predicted), max()), 5])
# axs.set_title('Stock')
# axs.set_ylabel('bvmf-pts')
# axs.grid(True)
# axs.autoscale(axis='x', tight=True)
# axs.plot(predicted[:, 0].astype(int), label='pred_max')
# axs.plot(predicted[:, 1].astype(int), label='pred_min')
# axs.plot(real_y[:, 0], label='real_max')
# axs.plot(real_y[:, 1], label='real_min')
# axs.legend()

# cursor = Cursor(axs)
# fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)

# plt.show()
a = 1

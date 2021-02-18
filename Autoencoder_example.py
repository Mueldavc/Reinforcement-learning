import keras
from keras import layers
from Data_treatment.Data_Treatment import StockData
from MetaTrader5 import TIMEFRAME_M5
from datetime import datetime
from keras.losses import MeanSquaredError
from keras.callbacks import TensorBoard


StockData = StockData("WINZ20", 10, 1)
StockData.download_stock(TIMEFRAME_M5, datetime.today(), 20)
StockData.values_norm()

# input_img = keras.Input(shape=(28, 28, 1))
input_img = keras.Input(shape=StockData.ENVIRONMENT_SHAPE)

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(2, (3, 3), activation='sigmoid', padding='same')(x)

loss_fn = MeanSquaredError()

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss=loss_fn)

autoencoder.fit(StockData.train_x, StockData.train_y,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_split=0.2,
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


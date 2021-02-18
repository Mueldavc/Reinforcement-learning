import numpy as np
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray
from Data_treatment.Data_Treatment import StockData_Bin
from datetime import datetime
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten

np.random.seed(1120)


def prepare_dataset(window_size, window_days):
    s = StockData_Bin("WIN$", window_size, 1)
    s.stockdata(5, datetime.today(), window_days)
    return s


def uint_conf(num):
    num = num.uint
    if isinstance(num, int):
        return num
    else:
        a = 1


def train_evaluate(ga_individual_solution):
    # Decode GA solution to integer for window_size and num_units
    window_size_bits = BitArray(ga_individual_solution[0:6])
    window_days = BitArray(ga_individual_solution[6:13])
    num_units_bits = BitArray(ga_individual_solution[13:17])
    num_units_bits_1 = BitArray(ga_individual_solution[17:21])
    epocas_bits = BitArray(ga_individual_solution[21:])

    window_size = uint_conf(window_size_bits)
    window_days = uint_conf(window_days)
    num_units = uint_conf(num_units_bits)
    num_units_1 = uint_conf(num_units_bits_1)
    epocas = uint_conf(epocas_bits)

    print('\nWindow Size: ', window_size,
          ', windows days:', window_days,
          ', Num of Units: ', num_units,
          ', num_units_1:', num_units_1,
          ', epocas:', epocas)

    # Return fitness score of 100 if window_size or num_unit is zero
    if any([0 in [window_days, num_units, epocas],
            window_days * 40 < window_size,
            window_size <= num_units,
            window_size <= 1]):
        return 0,

    # Segment the train_data based on new window_size; split into train and validation (80/20)
    try:
        stockdata = prepare_dataset(window_size, window_days)
    except:
        return 0,

    # X_train, X_val, y_train, y_val = split(stockdata.train_x, stockdata.train_y, test_size=0.20, random_state=1120)
    train_x = stockdata.train_x
    y_train = stockdata.train_y
    test_x = stockdata.x_today
    y_test = stockdata.y_today
    # X_train = X_train.reshape(X_train.shape[0], -1)
    # X_val = X_val.reshape(X_val.shape[0], -1)
    y_train = to_categorical(y_train)
    # Train LSTM model and predict on validation set
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)

    model = Sequential()

    if num_units_1 != 0:
        model.add(
            Conv2D(num_units, (3, 3), activation='relu', padding='same', input_shape=(*stockdata.ENVIRONMENT_SHAPE, 1)))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(num_units_1, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))

        model.add(Conv2D(num_units_1, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(num_units, (3, 3), activation='relu'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(2, (3, 3), activation='sigmoid', padding='same'))

    else:
        model.add(
            Conv2D(num_units, (3, 3), activation='relu', padding='same', input_shape=(*stockdata.ENVIRONMENT_SHAPE, 1)))
        model.add(MaxPooling2D((2, 2), padding='same'))

        model.add(Conv2D(num_units, (3, 3), activation='relu'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(2, (3, 3), activation='sigmoid', padding='same'))
    # if num_units_1 != 0:
    #     model.add(LSTM(num_units, return_sequences=True, input_shape=stockdata.ENVIRONMENT_SHAPE))
    #     model.add(LSTM(num_units_1))  # returns a sequence of vectors of dimension 32
    # else:
    #     model.add(LSTM(num_units, input_shape=stockdata.ENVIRONMENT_SHAPE))

    # model = Sequential()

    # if num_units_1 != 0:
    #     model.add(Dense(num_units, input_dim=X_train.shape[1], activation='relu'))
    #     model.add(Dense(num_units_1, activation='relu'))
    # else:
    #     model.add(Dense(num_units, input_dim=60, activation='relu'))

    # model.add(Dense(2, activation='linear')) binary_crossentropy
    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_x, y_train, epochs=epocas, batch_size=10, shuffle=True, verbose=2)
    y_pred = model.predict(test_x)
    y_pred = [np.argmax(i) for i in y_pred]

    # Calculate the RMSE score as fitness score for GA
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('Validation RMSE: ', rmse, '\n')

    return rmse,


population_size = 20
num_generations = 20
gene_length = 25

# As we are trying to minimize the RMSE score, that's why using -1.0.
# In case, when you want to maximize accuracy for instance, use 1.0
creator.create('FitnessMax', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('binary', bernoulli.rvs, 0.5)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n=gene_length)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

toolbox.register('mate', tools.cxOrdered)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.6)
toolbox.register('select', tools.selRoulette)
toolbox.register('evaluate', train_evaluate)

population = toolbox.population(n=population_size)
r = algorithms.eaSimple(population, toolbox, cxpb=0.4, mutpb=0.1, ngen=num_generations, verbose=False)

# Print top N solutions - (1st only, for now)
best_individuals = tools.selBest(population, k=1)
window_size = None
window_days = None
num_units = None
num_units_1 = None
epocas = None

for bi in best_individuals:
    window_size_bits = BitArray(bi[0:6])
    window_days = BitArray(bi[6:13])
    num_units_bits = BitArray(bi[13:17])
    num_units_bits_1 = BitArray(bi[17:21])
    epocas_bits = BitArray(bi[21:])

    window_size = uint_conf(window_size_bits)
    window_days = uint_conf(window_days)
    num_units = uint_conf(num_units_bits)
    num_units_1 = uint_conf(num_units_bits_1)
    epocas = uint_conf(epocas_bits)

    # if best_window_size != 0 and best_days_size != 0 and best_num_units != 0:
    #     break
print('Final\nWindow Size: ', window_size,
      ', windows days:', window_days,
      ', Num of Units: ', num_units,
      ', num_units_1:', num_units_1,
      ', epocas:', epocas)

# Train the model using best configuration on complete training set
# and make predictions on the test set
stockdata = prepare_dataset(window_size, window_days)
train_x = stockdata.train_x
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
train_y = to_categorical(stockdata.train_y)
val_y = stockdata.y_today
test_x = stockdata.x_today
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)
# inputs = Input(shape=stockdata.ENVIRONMENT_SHAPE)
# x = LSTM(best_num_units, input_shape=stockdata.ENVIRONMENT_SHAPE)(inputs)
# predictions = Dense(len(stockdata.y_cols), activation='linear')(x)
# model = Model(inputs=inputs, outputs=predictions)

model = Sequential()
if num_units_1 != 0:
    model.add(
        Conv2D(num_units, (3, 3), activation='relu', padding='same', input_shape=(*stockdata.ENVIRONMENT_SHAPE, 1)))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(num_units_1, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(num_units_1, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(num_units, (3, 3), activation='relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(2, (3, 3), activation='sigmoid', padding='same'))

else:
    model.add(
        Conv2D(num_units, (3, 3), activation='relu', padding='same', input_shape=(*stockdata.ENVIRONMENT_SHAPE, 1)))
    model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Dense(2, activation='sigmoid'))
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=epocas, batch_size=10, shuffle=True, verbose=2)
model.save(r'C:\Users\mueld\Documents\Python_Projects\Stock_1\model.h5')

y_pred = model.predict(test_x)
y_pred = [np.argmax(i) for i in y_pred]
rmse = np.sqrt(mean_squared_error(y_pred, val_y))
print('Test RMSE: ', rmse)

# predict = stockdata.scaler_y.inverse_transform(predict)
# predict = np.append(predict, [np.nan, np.nan]).reshape(-1, 2)
# real_y = np.append([np.nan, np.nan], stockdata.y_today).reshape(-1, 2)
# plt.title('Stock {}'.format(best_window_size))
# plt.ylabel('bvmf-pts')
# plt.grid(True)
# plt.autoscale(axis='x', tight=True)
# plt.plot(predict[:, 0], label='pred_max')
# plt.plot(predict[:, 1], label='pred_min')
# plt.plot(real_y[:, 0], label='real_max')
# plt.plot(real_y[:, 1], label='real_min')
# plt.legend()
# plt.show()

a = 1

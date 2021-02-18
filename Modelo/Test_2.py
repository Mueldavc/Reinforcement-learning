from Data_treatment.DT_1 import StockData

import numpy as np
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray
from datetime import datetime
from keras.utils import to_categorical
from keras.layers import Dense

np.random.seed(1120)

stockdata = StockData('WIN$', date=datetime.today(), window_days=1000, timeframe=5)


# stockdata = StockData('PETR4', date=datetime.today(), window_days=1000, timeframe=TIMEFRAME_H4)

def prepare_dataset(window_size, data_days, cols_selec, dump=False):
    stockdata.data_final(n_in=window_size, window_days=data_days, cols=cols_selec, dump=dump)
    return stockdata


def uint_conf(num):
    num = num.uint
    if isinstance(num, int):
        return num
    else:
        a = 1


def train_evaluate(ga_individual_solution):
    # Decode GA solution to integer for window_size and num_units
    window_size_bits = BitArray(ga_individual_solution[0:6])
    window_days = BitArray(ga_individual_solution[6:14])
    num_units_bits = BitArray(ga_individual_solution[14:18])
    num_units_bits_1 = BitArray(ga_individual_solution[18:22])
    epocas_bits = BitArray(ga_individual_solution[22:26])
    cols_selec = ga_individual_solution[26:]

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
            window_size <= 1,
            sum(cols_selec) < 2]):
        return 0,

    # Segment the train_data based on new window_size; split into train and validation (80/20)
    stockdata = prepare_dataset(window_size, window_days, cols_selec)
    print(stockdata.cols_ch)
    train_x = stockdata.train_x.values
    y_train = stockdata.train_y.values
    test_x = stockdata.val_x.values
    y_test = stockdata.val_y
    y_train = to_categorical(y_train)

    model = Sequential()

    if num_units_1 != 0:
        model.add(Dense(num_units, input_dim=train_x.shape[1], activation='relu'))
        model.add(Dense(num_units_1, activation='relu'))
    else:
        model.add(Dense(num_units, input_dim=train_x.shape[1], activation='relu'))

    model.add(Dense(2, activation='linear'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_x, y_train, epochs=epocas, batch_size=10, shuffle=True, verbose=2)
    y_pred = model.predict(test_x)
    y_pred = [np.argmax(i) for i in y_pred]

    # Calculate the RMSE score as fitness score for GA
    rmse = np.sqrt(mean_squared_error(y_test, y_pred[:-1]))
    print('Validation RMSE: ', rmse, '\n')

    return rmse,


population_size = 40
num_generations = 40
gene_length = 26 + 22

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
    epocas_bits = BitArray(bi[21:25])
    cols_selec = bi[25:]

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
stockdata = prepare_dataset(window_size, window_days, cols_selec, dump=True)
print(stockdata.cols_ch)
train_x = stockdata.train_x.values
y_train = stockdata.train_y.values
test_x = stockdata.val_x.values
y_test = stockdata.val_y
y_train = to_categorical(y_train)

model = Sequential()

if num_units_1 != 0:
    model.add(Dense(num_units, input_dim=train_x.shape[1], activation='relu'))
    model.add(Dense(num_units_1, activation='relu'))
else:
    model.add(Dense(num_units, input_dim=train_x.shape[1], activation='relu'))

model.add(Dense(2, activation='linear'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, y_train, epochs=epocas, batch_size=10, shuffle=True, verbose=2)
model.save(r'C:\Users\mueld\Documents\Python_Projects\Stock_1\model.h5')

y_pred = model.predict(test_x)
y_pred = [np.argmax(i) for i in y_pred]
rmse = np.sqrt(mean_squared_error(y_test, y_pred[:-1]))
print('Test RMSE: ', rmse)
print(cols_selec)
a = 1

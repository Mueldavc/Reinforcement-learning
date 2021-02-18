from Data_treatment.FinIndicator import Indicators
import pandas as pd
import MetaTrader5 as mt5
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np
import pickle


class StockData(Indicators):
    def __init__(self, stock, n_in=1, n_out=1):
        self.ACTION_SPACE = None
        self.ACTION_SPACE_SIZE = None
        self.ENVIRONMENT_SHAPE = None
        self.n_features = None
        self.n_in = n_in
        self.n_out = n_out
        self.n_obs = None
        self.stock = stock
        self.total_lines = None
        self.ticks_today = []
        self.ticks_frame = []
        self.train_x = None
        self.train_y = []
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self.x_today = None
        self.y_today = []
        self.real_y = None
        self.past_days = None
        self.stockastick_period = 10
        self.graf_x = []
        self.y_cols = None

    def stockdata(self, timeframe, date, past_days):
        """ Valores normalizados pelo dia"""
        self.past_days = past_days
        if not mt5.initialize():
            print("initialize() failed")
            mt5.shutdown()
        try:
            date - timedelta(days=past_days)
        except:
            past_days = past_days.uint

        ticks = mt5.copy_rates_range(self.stock, timeframe, date - timedelta(days=past_days), date)
        mt5.shutdown()

        tick_frame = pd.DataFrame(ticks)
        tick_frame.time = pd.to_datetime(tick_frame.time, unit='s')
        tick_frame = self.RSI(tick_frame, 2)
        tick_frame = self.stochastic(tick_frame, 10)
        tick_frame = tick_frame.set_index(tick_frame.time)
        tick_frame.drop(['time', 'spread', 'tick_volume'], axis='columns', inplace=True)
        self.y_cols = [-1 - list(reversed(tick_frame.columns)).index('high'),
                       -1 - list(reversed(tick_frame.columns)).index('low')]
        self.ticks_frame = tick_frame
        ticks = []
        periods = self.ticks_frame.index.to_period('d').drop_duplicates().sort_values().values
        for s, i in enumerate(periods):
            tick = self.ticks_frame.loc[self.ticks_frame.index.to_period('d') == i, :]
            tmp = tick.loc[:, ['high', 'low']]
            tmp.reset_index(inplace=True, drop=True)
            dif = ((tmp.high - tmp.low) * 0.1).astype(int)
            tmp['high_d'] = tmp.high - dif
            tmp['low_d'] = tmp.low + dif
            scaler_x = MinMaxScaler(feature_range=(0, 1))
            scaler_y = MinMaxScaler(feature_range=(0, 1))
            if s == len(periods) - 2:
                self.n_features = int(tick.shape[1])
                self.n_obs = int(tick.shape[1] * self.n_in)
                _scaler_y = scaler_y
                _scaler_x = scaler_x

            elif s >= len(periods) - 1:
                ind = tick.index.values
                tick = _scaler_x.fit_transform(tick.values)
                self.scaler_x = _scaler_x
                self.scaler_y.fit(tmp.loc[:, ['high_d', 'low_d']].values)
                tick = self._series_to_supervised(tick)
                pickle.dump(self.scaler_x,
                            open(r'C:\Users\mueld\Documents\Python_Projects\Stock_1\scaler_X.sav',
                                 'wb'))
                pickle.dump(self.scaler_y,
                            open(r'C:\Users\mueld\Documents\Python_Projects\Stock_1\scaler_Y.sav',
                                 'wb'))
                if not tick.empty:
                    self.ticks_today.append(tick)
                    self.y_today.append(tmp.iloc[-len(tick):,
                                        [tmp.columns.tolist().index('high_d'),
                                         tmp.columns.tolist().index('low_d')]].values[-len(tick):, :])
                    self.graf_x.append(ind[-len(tick):])
                else:
                    a = 1
                continue
            tick = scaler_x.fit_transform(tick.values.astype('float32'))
            tick = self._series_to_supervised(tick)
            if not tick.empty:
                tick.reset_index(inplace=True, drop=True)
                ticks.append(tick.values)
                _tmp = scaler_y.fit_transform(tmp.iloc[-len(tick):,
                                              [tmp.columns.tolist().index('high_d'),
                                               tmp.columns.tolist().index('low_d')]].values)
                self.train_y.append(_tmp)

        ticks = np.concatenate(ticks)
        # ticks = np.random.permutation(ticks)
        self.total_lines = len(ticks)
        self.train_y = np.concatenate(self.train_y)
        self.train_x = ticks[:, :self.n_obs].reshape(-1, self.n_features, self.n_in)
        self.ENVIRONMENT_SHAPE = (self.train_x.shape[1], self.train_x.shape[2])
        ticks = np.concatenate(self.ticks_today)
        self.y_today = np.concatenate(self.y_today)
        self.x_today = ticks[:, self.n_features:].reshape(-1, self.n_features, self.n_in)
        self.graf_x = np.concatenate(self.graf_x)
        if len(self.train_x) != len(self.train_y):
            a = 1

    def _series_to_supervised(self, data):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(self.n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, self.n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        agg.dropna(inplace=True)
        return agg

    def reset(self, step):
        return self.train_x[step - 1, :]

    def step(self, action, step):
        reward = ((self.train_y[step - 1] - action) / self.train_y[step - 1])
        return reward

    def new_action(self, step):
        population = [(random.random(), random.random()) for i in range(50)]
        population = np.array([[*i, *self.step(i, step)] for i in population])
        best_score = population[0, 2:]
        action = population[0, :2]
        for i in population:
            if all(i[2:] < best_score):
                best_score = i[2:]
                action = i[:2]
        return np.array(action)


class DecisionTreeeData(Indicators):
    def __init__(self, stock, window_days, timeframe, date):
        self.stock = stock
        self.window_days = window_days
        self.timeframe = timeframe
        self.date = date
        self._download()

    def _download(self):
        if not mt5.initialize():
            print("initialize() failed")
            mt5.shutdown()
        tick_frame = mt5.copy_rates_range(self.stock, self.timeframe, self.date - timedelta(days=self.window_days),
                                          self.date)
        mt5.shutdown()
        tick_frame = pd.DataFrame(tick_frame)
        tick_frame.time = pd.to_datetime(tick_frame.time, unit='s')
        tick_frame = self.RSI(tick_frame, 2)
        tick_frame = self.stochastic(tick_frame, 10)
        tick_frame = tick_frame.set_index(tick_frame.time)
        tick_frame.drop(['time', 'spread', 'tick_volume'], axis='columns', inplace=True)
        tick_frame['close_dif'] = tick_frame.close.diff(+1).apply(lambda x: 1 if x > 0 else 0).shift(-1).fillna(2)

        self.tick_frame_train = tick_frame.loc[tick_frame.index.date != self.date.date(), :].dropna()
        self.tick_frame_test = tick_frame.loc[tick_frame.index.date == self.date.date(), :]


class StockData_Bin(Indicators):
    def __init__(self, stock, n_in=1, n_out=1):
        self.ENVIRONMENT_SHAPE = None
        self.n_features = None
        self.n_in = n_in
        self.n_out = n_out
        self.n_obs = None
        self.stock = stock
        self.total_lines = None
        self.train_x = None
        self.train_y = None
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self.x_today = None
        self.past_days = None
        self.stockastick_period = 10

    def stockdata(self, timeframe, date, past_days):
        """ Valores normalizados pelo dia"""
        self.past_days = past_days
        if not mt5.initialize():
            print("initialize() failed")
            mt5.shutdown()
        date - timedelta(days=past_days)

        ticks = mt5.copy_rates_range(self.stock, timeframe, date - timedelta(days=past_days), date)
        mt5.shutdown()

        tick_frame = pd.DataFrame(ticks)
        tick_frame.time = pd.to_datetime(tick_frame.time, unit='s')
        tick_frame['close_dif'] = tick_frame.close - tick_frame.open
        tick_frame.close_dif = tick_frame.close_dif.apply(lambda x: 1 if x > 0 else 0).shift(-1).dropna()
        # tick_frame['close_dif'] = tick_frame.close.diff(+1).apply(lambda x: 1 if x > 0 else 0).shift(-1).fillna(0)
        tick_frame = self.RSI(tick_frame, 2)
        tick_frame = self.stochastic(tick_frame, 10)
        tick_frame = tick_frame.set_index(tick_frame.time)
        tick_frame.drop(['time', 'spread', 'tick_volume'], axis='columns', inplace=True)
        tick_frame.dropna(inplace=True)
        self.ticks_frame = tick_frame
        ticks = []
        periods = self.ticks_frame.index.to_period('d').drop_duplicates().sort_values().values
        for s, i in enumerate(periods):
            tick = self.ticks_frame.loc[
                self.ticks_frame.index.to_period('d') == i, ['open', 'high', 'low', 'close', 'real_volume',
                                                             'RSI', 'K']]
            if tick.empty:
                a=1
            scaler_x = MinMaxScaler(feature_range=(0, 1))
            tick = scaler_x.fit_transform(tick.values.astype('float32'))
            if s == len(periods) - 2:
                self.n_features = int(tick.shape[1])
                self.n_obs = int(tick.shape[1] * self.n_in)
                _scaler_x = scaler_x

            elif s >= len(periods) - 1:
                _scaler_x.fit(tick)
                self.scaler_x = _scaler_x
                pickle.dump(_scaler_x,
                            open(r'C:\Users\mueld\Documents\Python_Projects\Stock_1\scaler_X.sav',
                                 'wb'))
            if tick.size != 0:
                ticks.append(tick)

        ticks = np.concatenate(ticks)
        ticks = self._series_to_supervised(ticks)
        ticks = ticks.set_index(tick_frame.index[-ticks.shape[0]:])
        self.total_lines = len(ticks)
        self.train_x = ticks.iloc[ticks.index.date != date.date(),
                       :self.n_obs].values.reshape(-1, self.n_features, self.n_in)
        _ = tick_frame.close_dif[-ticks.shape[0]:]
        self.train_y = tick_frame.close_dif[-ticks.shape[0]:][ticks.index.date != date.date()].values
        self.x_today = ticks.iloc[ticks.index.date == date.date(),
                       :self.n_obs].values.reshape(-1, self.n_features, self.n_in)

        self.ENVIRONMENT_SHAPE = (self.train_x.shape[1], self.train_x.shape[2])

        self.y_today = tick_frame.close_dif[tick_frame.index.date == date.date()].values
        self.x_today = ticks.iloc[ticks.index.date == date.date(),
                       self.n_features:].values.reshape(-1, self.n_features, self.n_in)
        a=1

    def _series_to_supervised(self, data):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(self.n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, self.n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        agg.dropna(inplace=True)
        return agg

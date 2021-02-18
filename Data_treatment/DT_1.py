import MetaTrader5 as mt5
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator, stochrsi_k, ultimate_oscillator
from ta.volume import volume_price_trend, VolumeWeightedAveragePrice


class StockData(object):
    def __init__(self, stock, date, window_days, timeframe):
        self.stock = stock
        self.date = date
        self.window_days = window_days
        self.timeframe = timeframe
        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self._Y = None
        self.val_y = None
        self._download()

    def _download(self):
        if not mt5.initialize():
            print("initialize() failed")
            mt5.shutdown()
        ctrl = 0
        date_ini = self.date - timedelta(days=1)
        date_fim = self.date
        _ = []
        while True:
            tick_frame = mt5.copy_rates_range(self.stock,
                                              self.timeframe,
                                              date_ini,
                                              date_fim)
            tick_frame = pd.DataFrame(tick_frame)
            date_ini -= timedelta(days=1)
            date_fim -= timedelta(days=1)
            if not tick_frame.empty:
                _.append(tick_frame)
                ctrl += 1
            if ctrl == self.window_days:
                tick_frame = pd.concat(_)
                break
        tick_frame.time = pd.to_datetime(tick_frame.time, unit='s')
        tick_frame.set_index(tick_frame.time, inplace=True)
        tick_frame.sort_index(inplace=True)
        tick_frame.drop_duplicates(inplace=True)

        indicator_bb = BollingerBands(close=tick_frame.close, window=10, window_dev=2)
        indicator_rsi = RSIIndicator(close=tick_frame.close, window=2).rsi()
        indicator_stk = stochrsi_k(close=tick_frame.close, window=10)
        indicator_vol = volume_price_trend(close=tick_frame.close, volume=tick_frame.real_volume)
        indicator_vwap = VolumeWeightedAveragePrice(close=tick_frame.close,
                                                    volume=tick_frame.real_volume,
                                                    high=tick_frame.high,
                                                    low=tick_frame.low)
        indicator_uo = ultimate_oscillator(low=tick_frame.low, close=tick_frame.close, high=tick_frame.high)
        tick_frame['bb_bbm'] = indicator_bb.bollinger_mavg()
        tick_frame['bb_bbh'] = indicator_bb.bollinger_hband()
        tick_frame['bb_bbl'] = indicator_bb.bollinger_lband()
        tick_frame['rsi'] = indicator_rsi
        tick_frame['vol'] = indicator_vol
        tick_frame['k'] = indicator_stk
        tick_frame['vwap'] = indicator_vwap.vwap
        tick_frame['uo'] = indicator_uo

        close_dif = tick_frame.close - tick_frame.open
        self._Y = close_dif.apply(lambda x: 1 if x > 0 else 0).shift(-1).dropna()
        tick = pd.concat([tick_frame.loc[:, ['low', 'open', 'close', 'high']].shift(1),
                          tick_frame.loc[:, ['low', 'open', 'close', 'high']]], axis=1).dropna().drop_duplicates()
        tick = tick.apply(lambda x: self._df_func(x), axis=1)

        pkl.dump(tick, open(
            r'C:\Users\mueld\Documents\Python_Projects\reinforcement-learning-using-python-master\Data_treatment\DataBase\datazone.pandas',
            'wb'))

        tick = pd.concat(
            [tick, tick_frame.loc[:, ['open', 'high', 'low', 'close', 'tick_volume', 'real_volume', 'bb_bbm', 'bb_bbh',
                                      'bb_bbl',
                                      'rsi', 'vol', 'k', 'vwap', 'uo']]], axis=1)
        tick.dropna(inplace=True)
        # s = ['low_l', 'low_h', 'open_l', 'open_h', 'close_l', 'close_h', 'high_l', 'high_h', 'bb_bbm', 'bb_bbh',
        #      'bb_bbl',
        #      'rsi', 'vol', 'k', 'vwap']
        self.tickframe = tick

    def data_final(self, cols, n_in=1, n_out=1, window_days=1, dump=False):
        self.cols_ch = [__ for _, __ in zip(cols, self.tickframe.columns) if _ == 1]
        timeframe = self.tickframe.loc[:, [__ for _, __ in zip(cols, self.tickframe.columns) if _ == 1]]
        ticks = self._series_to_supervised(timeframe, n_in, n_out, dump=dump)
        periods = ticks.index.to_period('d').drop_duplicates().sort_values(ascending=False).values[:window_days + 1]
        self.train_x = ticks.loc[(ticks.index.to_period('d') != periods[0]) &
                                 ticks.index.to_period('d').isin(periods), :]
        self.train_y = self._Y[self._Y.index.isin(self.train_x.index)]
        self.val_x = ticks.loc[ticks.index.to_period('d') == periods[0], :]
        self.val_y = self._Y[self._Y.index.to_period('d') == periods[0]]

    def _series_to_supervised(self, tickframe, n_in, n_out, dump=False):
        data = self.scaler_x.fit_transform(tickframe)
        if dump:
            pkl.dump(self.scaler_x,
                     open(r'C:\Users\mueld\Documents\Python_Projects\Stock_1\scaler_X.sav',
                          'wb'))
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.set_index(self.tickframe.index, inplace=True)
        agg.columns = names
        # drop rows with NaN values
        agg.dropna(inplace=True)
        return agg

    def _df_func(self, x):
        x.reset_index(inplace=True, drop=True)
        s = []
        for i in range(4, 8):
            h = x[i] > x[0:4]
            l = x[i] < x[0:4]
            h = h[h].index.max() + 1 if not np.isnan(h[h].index.max()) else 0
            l = l[l].index.min() if not np.isnan(l[l].index.max()) else 0
            s.append([h, l])
        return pd.Series(np.reshape(s, -1),
                         index=['low_l', 'low_h', 'open_l', 'open_h', 'close_l', 'close_h', 'high_l', 'high_h'])

    def __del__(self):
        mt5.shutdown()

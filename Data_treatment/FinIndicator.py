class Indicators(object):
    def RSI(self, stock, period=14):
        # Wilder's RSI
        close = stock['close']
        delta = close.diff()
        up, down = delta.copy(), delta.copy()

        up[up < 0] = 0
        down[down > 0] = 0

        # Calculate the exponential moving averages (EWMA)
        roll_up = up.ewm(com=period - 1, adjust=False).mean()
        roll_down = down.ewm(com=period - 1, adjust=False).mean().abs()

        # Calculate RS based on exponential moving average (EWMA)
        rs = roll_up / roll_down  # relative strength =  average gain/average loss

        rsi = 100 - (100 / (1 + rs))
        stock['RSI'] = rsi
        return stock

    def stochastic(self, df, window):
        L_w = df['low'].rolling(window=window).min()
        H_w = df['high'].rolling(window=window).max()
        df['K'] = 100 * ((df['close'] - L_w) / (H_w - L_w))
        return df

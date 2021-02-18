from numpy import unique
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from Data_treatment.Data_Treatment import StockData_1
import matplotlib._color_data as mcd
from matplotlib.dates import date2num, DateFormatter
from random import seed, sample
import pandas as pd
from datetime import datetime

# define dataset
# X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
#                            random_state=4)

seed(1)
_ = list(mcd.CSS4_COLORS)
__ = ['aliceblue', 'beige']
for i in __:
    _.remove(i)
colors = sample(_, 14)

StockData = StockData_1("WIN$", 10, 1)
StockData.stockdata(5, datetime.today(), 600)

X = StockData.train_x.reshape(StockData.train_x.shape[0], -1)
x_test = StockData.train_x_today.reshape(StockData.train_x_today.shape[0], -1)
y_test = StockData.train_y_today

# define the model
# model = AffinityPropagation()
model = KMeans(n_clusters=14)  # k=34/1126 | k=14/1943 | k=7/2367
# model = Birch(threshold=0.01)  # k=28/1236 | k=11/1624
# model = AgglomerativeClustering()  # k=18/1887 | k=7/2443
# model = MiniBatchKMeans()  # k=35/1543 | k=11/2101

###### Elbow visualizer
#
# visualizer = KElbowVisualizer(model, k=(4, 40))
#
# visualizer.fit(X)  # Fit the data to the visualizer
# visualizer.show()  # Finalize and render the figure
############################################


# fit model and predict clusters
model.fit(X)
yhat = model.predict(x_test)
# yhat = list(reversed(yhat))
# retrieve unique clusters
clusters = unique(yhat)

fig, ax = plt.subplots()

# scatter = ax.scatter([i for i in range(yhat.size)], y[:, 0], c=yhat, s=10, cmap="Spectral")
# scatter = ax.scatter([i for i in range(yhat.size)], y[:, 1], c=yhat, s=10, cmap="Spectral")
# legend = ax.legend(*scatter.legend_elements(num=7), loc="upper left", title="Cluster")

pltframe = pd.DataFrame(yhat, columns=['yhat'])
pltframe['high'] = y_test[:, 0]
pltframe['low'] = y_test[:, 1]
pltframe = pltframe.set_index(StockData.graf_x)
pltframe['ind'] = StockData.graf_x
periods = pltframe.index.to_period('d').drop_duplicates().sort_values().values

fig, axs = plt.subplots(len(periods), sharex=True)

for __, _ in enumerate(periods):
    tick = pltframe.loc[pltframe.index.to_period('d') == _, :]
    s = -1
    lin2d = []
    mx = 0
    ini = 0
    for i, h in enumerate(tick.iterrows()):
        # lin2d.append(datetime.strptime("{a}:{b}:00".format(a=h[0].hour, b=h[0].minute), "%H:%M:%S"))
        if s != h[1].yhat:
            axs[__].plot(date2num(lin2d), tick.high[ini:i], c=colors[s])
            axs[__].plot(date2num(lin2d), tick.low[ini:i], c=colors[s])
            lin2d = []
            ini = i
        s = h[1].yhat
        lin2d.append(datetime.strptime("{a}:{b}:00".format(a=h[0].hour, b=h[0].minute), "%H:%M:%S"))
    axs[__].xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    axs[__].set_title(_)
fig.tight_layout()
fig.show()

s = -1
lin2d = []
mx = 0
ini = 0
# y = StockData.scaler_y.inverse_transform(y)
for i, h in enumerate(yhat):
    lin2d.append(i)
    if s != h:
        ax.plot(lin2d, y_test[ini:i, 0], c=colors[s])
        ax.plot(lin2d, y_test[ini:i, 1], c=colors[s])
        lin2d = []
        ini = i

    s = h

# ax.add_artist(legend)
# plt.legend()
ax.grid(True)
plt.show()
a = 1

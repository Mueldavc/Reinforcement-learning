from random import choice
from collections import deque
import time
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tqdm import tqdm
import random
import os

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

PATH = r'D:\PYTHON\reinforcement-learning-using-python-master'
# Create models folder
if not os.path.isdir(r'{}models_s'.format(PATH)):
    os.makedirs(r'{}models_s'.format(PATH))
# Create results folder
if not os.path.isdir(r'{}results_s'.format(PATH)):
    os.makedirs(r'{}results_s'.format(PATH))

TstartTime = time.time()

"""Importação de dados"""
######################################################################################

import MetaTrader5 as mt5
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler


class data_class:
    F_HEIGHT = 14
    WIDTH = 10
    ENVIRONMENT_SHAPE = (int(F_HEIGHT / 2), WIDTH, 1)
    ACTION_SPACE = [0, 1]
    ACTION_SPACE_SIZE = len(ACTION_SPACE)
    PUNISHMENT = -100  # Punishment increment
    REWARD = 10  # Reward increment
    score = 0  # Initial Score

    MOVE_WALL_EVERY = 4  # Every how many frames the wall moves.
    MOVE_PLAYER_EVERY = 1  # Every how many frames the player moves.
    frames_counter = 0

    def __init__(self, n_obs, n_features, n_in=1, n_out=1, dropnan=True):
        self.n_in = n_in
        self.n_out = n_out
        self.dropnan = dropnan
        self.n_obs = n_obs
        self.n_features = n_features
        self.replay_memory = deque(maxlen=5)
        self.total_lines = 0

    def values_xy(self, reframed):
        train = reframed.values
        train_X = train[:, :self.n_obs]
        train_x = train_X.reshape(train_X.shape[0], self.F_HEIGHT, self.WIDTH)
        train_y = train[:, [-4, -3]]
        self.total_lines += train_x.shape[0]
        return train_X, train_y

    def reset(self, step):
        self.game_over = False
        return self.total_x[step - 1, :]

    def step(self, action, step):
        reward = np.average((self.total_y[step - 1] - action) / self.total_y[step - 1])
        if abs(reward) <= 0.05 or step == 15:
            game_over = True
        else:
            game_over = self.game_over
        return abs(reward), game_over, self.total_y[step - 1]

    def update_replay_memory(self, data, transition):
        train_x, train_y = self.values_xy(self.series_to_supervised(data))
        self.replay_memory.append((train_x, train_y, *transition))

    def series_to_supervised(self, data):
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
        if self.dropnan:
            agg.dropna(inplace=True)
        return agg

    def total_data(self):
        self.total_x = np.concatenate([i[0] for i in self.replay_memory])
        self.total_y = np.concatenate([i[1] for i in self.replay_memory])


# convert series to supervised learning

if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

today = datetime.today()
n_min = 20
n_features = 7
n_obs = n_min * n_features
dc = data_class(n_obs, n_features, n_min, 1)
cont = True
day = 0
ctr_day = 0
while cont:
    win_ticks_today = mt5.copy_rates_range("WINZ20", mt5.TIMEFRAME_M5,
                                           datetime(today.year, today.month, today.day, 6, 0) - timedelta(days=day),
                                           datetime(today.year, today.month, today.day, 16, 30) - timedelta(days=day))

    if win_ticks_today.size != 0:
        ticks_frame = pd.DataFrame(win_ticks_today)
        ticks_frame.time = pd.to_datetime(ticks_frame.time, unit='s')
        ticks_frame['h'] = ticks_frame.time.apply(lambda x: x.hour)
        ticks_frame['m'] = ticks_frame.time.apply(lambda x: x.minute)
        ticks_frame = ticks_frame[['h', 'm', 'open', 'high', 'low', 'close', 'real_volume']]
        values = ticks_frame.values
        values = values.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        dc.update_replay_memory(scaled, (scaler, ctr_day))
        ctr_day += 1
    day += 1
    if ctr_day == 5:
        cont = False

mt5.shutdown()
dc.total_data()


class ModifiedTensorBoard(TensorBoard):
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, name)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()


######################################################################################
# Agent class

class DQNAgent:
    def __init__(self, name, data_class, conv_list, dense_list, util_list):
        self.dc = data_class
        self.conv_list = conv_list
        self.dense_list = dense_list
        self.name = [str(name) + "_" + "".join(str(c) + "C_" for c in conv_list) + "".join(
            str(d) + "D_" for d in dense_list) + "".join(u + "_" for u in util_list)][0]

        # Main model
        self.model = self.create_model(self.conv_list, self.dense_list)

        # Target network
        self.target_model = self.create_model(self.conv_list, self.dense_list)
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(name, log_dir="{}logs_s/{}-{}".format(PATH, name, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    # Creates a convolutional block given (filters) number of filters, (dropout) dropout rate,
    # (bn) a boolean variable indecating the use of BatchNormalization,
    # (pool) a boolean variable indecating the use of MaxPooling2D 
    def conv_block(self, inp, filters=64, bn=True, pool=True, dropout=0.2):
        _ = Conv2D(filters=filters, kernel_size=3, activation='relu')(inp)
        if bn:
            _ = BatchNormalization()(_)
        if pool:
            _ = MaxPooling2D(pool_size=(2, 2))(_)
        if dropout > 0:
            _ = Dropout(0.2)(_)
        return _

    # Creates the model with the given specifications:
    def create_model(self, conv_list, dense_list):
        # Defines the input layer with shape = ENVIRONMENT_SHAPE
        input_layer = Input(shape=self.dc.ENVIRONMENT_SHAPE)
        # Defines the first convolutional block:
        _ = self.conv_block(input_layer, filters=conv_list[0], bn=False, pool=False)
        # If number of convolutional layers is 2 or more, use a loop to create them.
        if len(conv_list) > 1:
            for c in conv_list[1:]:
                _ = self.conv_block(_, filters=c)
        # Flatten the output of the last convolutional layer.
        _ = Flatten()(_)

        # Creating the dense layers:
        for d in dense_list:
            _ = Dense(units=d, activation='relu')(_)
        # The output layer has 5 nodes (one node per action)
        output = Dense(units=self.dc.ACTION_SPACE_SIZE,
                       activation='linear', name='output')(_)
        # Put it all together:
        model = Model(inputs=input_layer, outputs=[output])
        model.compile(optimizer=Adam(lr=0.001),
                      loss={'output': 'mse'},
                      metrics={'output': 'accuracy'})

        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states.reshape(-1, *dc.ENVIRONMENT_SHAPE))

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states.reshape(-1, *dc.ENVIRONMENT_SHAPE))

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            # if not done:
            #     max_future_q = future_qs_list[index]
            #     new_q = reward + DISCOUNT * max_future_q
            # else:
            #     new_q = reward

            # Update Q value for given state
            # current_qs = current_qs_list[index]
            # current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(action)

            # Fit on all samples as one batch, log only on terminal state
        self.model.fit(x=np.array(X).reshape(-1, *dc.ENVIRONMENT_SHAPE),
                       y=np.array(y),
                       batch_size=MINIBATCH_SIZE, verbose=0,
                       shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(state.reshape(-1, *dc.ENVIRONMENT_SHAPE))
        # return self.model.predict(state.reshape(-1, *data_class.x.shape))


######################################################################################
def save_model_and_weights(agent, model_name, episode, max_reward, average_reward, min_reward):
    checkpoint_name = r"{}_Eps({})_max({:7.2f})_avg({:7.2f})_min({:7.2f}).model".format(
        model_name, episode, max_reward, average_reward, min_reward)
    agent.model.save(r'{}models_s/{}'.format(PATH, checkpoint_name))
    best_weights = agent.model.get_weights()
    return best_weights


######################################################################################
# ## Constants:
# RL Constants:
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 1500  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 500  # Minimum number of steps in a memory to start training
UPDATE_TARGET_EVERY = 20  # Terminal states (end of episodes)
MIN_REWARD = 1000  # For model save
SAVE_MODEL_EVERY = 500  # Episodes
SHOW_EVERY = 20  # Episodes
EPISODES = 100  # Number of episodes
#  Stats settings
AGGREGATE_STATS_EVERY = 4  # episodes
SHOW_PREVIEW = False
######################################################################################
# Models Arch :
# [{[conv_list], [dense_list], [util_list], MINIBATCH_SIZE, {EF_Settings}, {ECC_Settings}} ]

models_arch = [{"conv_list": [32], "dense_list": [32, 32], "util_list": ["ECC2", "1A-5Ac"],
                "MINIBATCH_SIZE": 128, "best_only": False,
                "EF_Settings": {"EF_Enabled": False}, "ECC_Settings": {"ECC_Enabled": False}},

               {"conv_list": [32], "dense_list": [32, 32, 32], "util_list": ["ECC2", "1A-5Ac"],
                "MINIBATCH_SIZE": 128, "best_only": False,
                "EF_Settings": {"EF_Enabled": False}, "ECC_Settings": {"ECC_Enabled": False}},

               {"conv_list": [32], "dense_list": [32, 32], "util_list": ["ECC2", "1A-5Ac"],
                "MINIBATCH_SIZE": 128, "best_only": False,
                "EF_Settings": {"EF_Enabled": True, "FLUCTUATIONS": 2},
                "ECC_Settings": {"ECC_Enabled": True, "MAX_EPS_NO_INC": int(EPISODES * 0.2)}}]

# A dataframe used to store grid search results
res = pd.DataFrame(columns=["Model Name", "Convolution Layers", "Dense Layers", "Batch Size", "ECC", "EF",
                            "Best Only", "Average Reward", "Best Average", "Epsilon 4 Best Average",
                            "Best Average On", "Max Reward", "Epsilon 4 Max Reward", "Max Reward On",
                            "Total Training Time (min)", "Time Per Episode (sec)"])
######################################################################################
# Grid Search:

result = []
for i, m in enumerate(models_arch):
    startTime = time.time()  # Used to count episode training time
    MINIBATCH_SIZE = m["MINIBATCH_SIZE"]

    # Exploration settings :
    # Epsilon Fluctuation (EF):
    EF_Enabled = m["EF_Settings"]["EF_Enabled"]  # Enable Epsilon Fluctuation
    MAX_EPSILON = 1  # Maximum epsilon value
    MIN_EPSILON = 0.001  # Minimum epsilon value
    if EF_Enabled:
        FLUCTUATIONS = m["EF_Settings"]["FLUCTUATIONS"]  # How many times epsilon will fluctuate
        FLUCTUATE_EVERY = int(EPISODES / FLUCTUATIONS)  # Episodes
        EPSILON_DECAY = MAX_EPSILON - (MAX_EPSILON / FLUCTUATE_EVERY)
        epsilon = 1  # not a constant, going to be decayed
    else:
        EPSILON_DECAY = MAX_EPSILON - (MAX_EPSILON / (0.8 * EPISODES))
        epsilon = 1  # not a constant, going to be decayed

    # Initialize some variables: 
    best_average = 1
    best_score = 1

    # Epsilon Conditional Constantation (ECC):
    ECC_Enabled = m["ECC_Settings"]["ECC_Enabled"]
    avg_reward_info = [
        [1, best_average, epsilon]]  # [[episode1, reward1 , epsilon1] ... [episode_n, reward_n , epsilon_n]]
    max_reward_info = [[1, best_score, epsilon]]
    if ECC_Enabled: MAX_EPS_NO_INC = m["ECC_Settings"][
        "MAX_EPS_NO_INC"]  # Maximum number of episodes without any increment in reward average
    eps_no_inc_counter = 0  # Counts episodes with no increment in reward

    # For stats
    ep_rewards = [best_average]

    agent = DQNAgent(r"M{}".format(i), dc, m["conv_list"], m["dense_list"], m["util_list"])
    MODEL_NAME = agent.name

    best_weights = [agent.model.get_weights()]

    # Uncomment these two lines if you want to show preview on your screen
    # WINDOW          = pygame.display.set_mode((env.WINDOW_WIDTH, env.WINDOW_HEIGHT))
    # clock           = pygame.time.Clock()

    # Iterate over episodes
    for episode in tqdm(range(1, dc.total_lines + 1), ascii=True, unit='episodes'):
        if m["best_only"]: agent.model.set_weights(best_weights[0])
        # agent.target_model.set_weights(best_weights[0])

        score_increased = False
        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1
        action = 0
        # Reset data and get initial state
        current = dc.reset(episode)
        current_state = current[:int(current.shape[0] / 2)]
        game_over = dc.game_over

        while not game_over:
            # This part stays mostly the same, the change is to query a model for Q values
            # st = False
            if np.random.random() > epsilon:
                # Get action from Q table
                action = agent.get_qs(current_state).reshape(-1, )
                # st = True

            else:
                # Get random action
                action = (random.random(), random.random())

            reward, game_over, real = dc.step(action, step)
            new_state = current[int(current.shape[0] / 2):]
            # if st:
            #     print(reward)
            # Transform new continuous state to new discrete state and count reward
            episode_reward += reward

            # Uncomment the next block if you want to show preview on your screen
            # if SHOW_PREVIEW and not episode % SHOW_EVERY:
            #     clock.tick(27)
            #     env.render(WINDOW)

            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state, action, reward, new_state, game_over))
            agent.train(game_over, step)

            # current_state = new_state
            step += 1
        val = agent.get_qs(current_state).reshape(-1, )
        result.append([i, *val, *dc.total_y[episode]])
        if ECC_Enabled: eps_no_inc_counter += 1
        # Append episode reward to a list and log stats (every given number of episodes)
        episode_reward = np.average(episode_reward)
        ep_rewards.append(episode_reward)

        if not episode % AGGREGATE_STATS_EVERY:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=epsilon)

            # Save models, but only when avg reward is greater or equal a set value
            if not episode % SAVE_MODEL_EVERY:
                # Save Agent :
                _ = save_model_and_weights(agent, MODEL_NAME, episode, max_reward, average_reward, min_reward)

            if average_reward < best_average:
                best_average = average_reward
                # update ECC variables:
                avg_reward_info.append([episode, best_average, epsilon])
                eps_no_inc_counter = 0
                # Save Agent :
                best_weights[0] = save_model_and_weights(agent, MODEL_NAME.strip(), episode, max_reward, average_reward,
                                                         min_reward)

            if ECC_Enabled and eps_no_inc_counter >= MAX_EPS_NO_INC:
                epsilon = avg_reward_info[-1][2]  # Get epsilon value of the last best reward
                eps_no_inc_counter = 0

        if episode_reward < best_score:
            try:
                best_score = episode_reward
                max_reward_info.append([episode, best_score, epsilon])

                # Save Agent :
                best_weights[0] = save_model_and_weights(agent, MODEL_NAME.strip(), episode, max_reward, average_reward,
                                                         min_reward)

            except:
                pass

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

        # Epsilon Fluctuation:
        if EF_Enabled:
            if not episode % FLUCTUATE_EVERY:
                epsilon = MAX_EPSILON

    endTime = time.time()
    total_train_time_sec = round((endTime - startTime))
    total_train_time_min = round((endTime - startTime) / 60, 2)
    time_per_episode_sec = round((total_train_time_sec) / EPISODES, 3)

    # Get Average reward:
    average_reward = round(sum(ep_rewards) / len(ep_rewards), 2)

    # Update Results DataFrames:
    res = res.append({"Model Name": MODEL_NAME, "Convolution Layers": m["conv_list"], "Dense Layers": m["dense_list"],
                      "Batch Size": m["MINIBATCH_SIZE"], "ECC": m["ECC_Settings"], "EF": m["EF_Settings"],
                      "Best Only": m["best_only"], "Average Reward": average_reward,
                      "Best Average": avg_reward_info[-1][1], "Epsilon 4 Best Average": avg_reward_info[-1][2],
                      "Best Average On": avg_reward_info[-1][0], "Max Reward": max_reward_info[-1][1],
                      "Epsilon 4 Max Reward": max_reward_info[-1][2], "Max Reward On": max_reward_info[-1][0],
                      "Total Training Time (min)": total_train_time_min, "Time Per Episode (sec)": time_per_episode_sec}
                     , ignore_index=True)
    res = res.sort_values(by='Best Average')
    avg_df = pd.DataFrame(data=avg_reward_info, columns=["Episode", "Average Reward", "Epsilon"])
    max_df = pd.DataFrame(data=max_reward_info, columns=["Episode", "Max Reward", "Epsilon"])

    # Save dataFrames
    res.to_csv(r"{}results_s/Results.csv".format(PATH))
    avg_df.to_csv(r"{}results_s/{}-Results-Avg.csv".format(PATH, MODEL_NAME))
    max_df.to_csv(r"{}results_s/{}-Results-Max.csv".format(PATH, MODEL_NAME))

TendTime = time.time()
######################################################################################
print(r"Training took {} Minutes".format(round((TendTime - TstartTime) / 60)))
print(r"Training took {} Hours".format(round((TendTime - TstartTime) / 3600)))
######################################################################################

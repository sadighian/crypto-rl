from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, CuDNNLSTM, Dropout, MaxPooling1D, GlobalAveragePooling1D, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from .trading_gym import TradingGym


class DqnAgent(TradingGym):

    def __init__(self, step_size=4, window_length=4, train=True, training_steps=10000, weights=None):
        super(DqnAgent, self).__init__(step_size=step_size)
        self.window_length = window_length  # number of lags
        self.model = self.create_model()
        self.memory = SequentialMemory(limit=20000, window_length=self.window_length)
        self.train = train
        self.training_steps = training_steps
        self.weights = weights

        # create the agent
        self.agent = DQNAgent(model=self.model,
                              nb_actions=self.action_space.n,
                              memory=self.memory,
                              processor=None,
                              nb_steps_warmup=5000,
                              enable_dueling_network=True,  # enables double-dueling q-network
                              dueling_type='avg',
                              enable_double_dqn=True,
                              gamma=0.99,
                              target_model_update=1000,
                              delta_clip=1.0)

        self.agent.compile(Adam(lr=0.00048), metrics=['mae'])

    def create_model(self):
        """
        Create a neural network

        This class is intended to be modified by users if
        they are performing research on network architecture.
        :return: keras model
        """
        features_shape = (self.window_length, self.observation_space.shape[1])
        model = Sequential()

        model.add(CuDNNLSTM(256, input_shape=features_shape))#, return_sequences=True))
        model.add(Activation('relu'))

        model.add(Dense(self.action_space.n))
        model.add(Activation('linear'))

        model.add(Dense(self.action_space.n))
        model.add(Activation('softmax'))

        print(model.summary())
        return model

    def create_model2(self):
        features_shape = (self.window_length, self.observation_space.shape[1])

        model = Sequential()
        model.add(Conv1D(16, input_shape=features_shape, kernel_size=8, padding='same', activation='relu', strides=4))
        # model.add(MaxPooling1D(pool_size=3))
        # model.add(Dropout(0.3))
        model.add(Conv1D(32, kernel_size=4, padding='same', activation='relu', strides=2))
        # model.add(MaxPooling1D(pool_size=3))
        # model.add(Dropout(0.35))
        model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu', strides=1))
        # model.add(MaxPooling1D(pool_size=3))
        # model.add(Dropout(0.1))
        model.add(CuDNNLSTM(256))
        # model.add(Flatten())

        model.add(Dense(self.action_space.n))
        model.add(Activation('linear'))

        model.add(Dense(self.action_space.n))
        model.add(Activation('softmax'))

        print(model.summary())
        return model

    def start(self):
        if self.train:
            weights_filename = 'dqn_{}_weights.h5f'.format(self.env_id)
            checkpoint_weights_filename = 'dqn_' + self.env_id + '_weights_{step}.h5f'
            log_filename = 'dqn_{}_log.json'.format(self.env_id)

            callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
            callbacks += [FileLogger(log_filename, interval=100)]

            self.agent.fit(self, callbacks=callbacks, nb_steps=self.training_steps, log_interval=10000)
            self.agent.save_weights(weights_filename, overwrite=True)
            self.agent.test(self, nb_episodes=3, visualize=False)
        else:
            weights_filename = self.weights if self.weights else 'dqn_{}_weights.h5f'.format(self.env_id)
            self.agent.load_weights(weights_filename)
            self.agent.test(self, nb_episodes=3, visualize=True)


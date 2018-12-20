from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, CuDNNLSTM, Dropout, MaxPooling1D, GlobalAveragePooling1D, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from trading_gym import TradingGym


class DqnAgent(TradingGym):

    def __init__(self):
        super(DqnAgent, self).__init__(step_size=4)
        self.window_length = 8
        self.model = self.create_model3()
        self.memory = SequentialMemory(limit=20000, window_length=self.window_length)
        self.args = {
            'mode': 'train',
            'env_name': self.env_id,
            'weights': None
        }

        # create the agent
        self.agent = DQNAgent(model=self.model,
                              nb_actions=self.action_space.n,
                              memory=self.memory,
                              processor=None,
                              nb_steps_warmup=5000,
                              enable_dueling_network=True,
                              dueling_type='avg',
                              enable_double_dqn=True,
                              gamma=0.9999,
                              target_model_update=1000,
                              delta_clip=1.0)

        self.agent.compile(Adam(lr=1e-3), metrics=['mae'])

    def create_model(self):
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
        model.add(Conv1D(64, 3, activation='relu', input_shape=features_shape))
        # model.add(Conv1D(64, 3, activation='relu'))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(128, 3, activation='relu'))
        # model.add(Conv1D(128, 3, activation='relu'))
        # model.add(GlobalAveragePooling1D())

        # model.add(Dense(512))
        # model.add(Activation('relu'))

        model.add(CuDNNLSTM(256, return_sequences=True))
        model.add(Activation('relu'))

        model.add(Dropout(0.2))

        model.add(CuDNNLSTM(128))
        model.add(Activation('relu'))

        model.add(Dense(self.action_space.n))
        model.add(Activation('softmax'))

        print(model.summary())
        return model

    def create_model3(self):
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
        if self.args['mode'] == 'train':
            # Okay, now it's time to learn something! We capture the interrupt exception so that training
            # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
            weights_filename = 'dqn_{}_weights.h5f'.format(self.args['env_name'])
            checkpoint_weights_filename = 'dqn_' + self.args['env_name'] + '_weights_{step}.h5f'
            log_filename = 'dqn_{}_log.json'.format(self.args['env_name'])

            callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
            callbacks += [FileLogger(log_filename, interval=100)]

            self.agent.fit(self, callbacks=callbacks, nb_steps=40000, log_interval=10000)

            # After training is done, we save the final weights one more time.
            self.agent.save_weights(weights_filename, overwrite=True)

            # Finally, evaluate our algorithm for N episodes.
            self.agent.test(self, nb_episodes=3, visualize=False)
        elif self.args['mode'] == 'test':
            weights_filename = 'dqn_{}_weights.h5f'.format(self.args['env_name'])
            if self.args.weights:
                weights_filename = self.args['weights']
            self.agent.load_weights(weights_filename)
            self.agent.test(self, nb_episodes=3, visualize=True)


if __name__ == '__main__':
    print('training...')
    agent = DqnAgent()
    agent.start()
    print('...done training.')

import argparse

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv1D, CuDNNLSTM, Dropout, MaxPooling1D
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from trading_gym import TradingGym


class DqnAgent(TradingGym):

    def __init__(self):
        super(DqnAgent, self).__init__(step_size=4)
        self.window_length = 4
        self.model = self.create_model()
        self.memory = SequentialMemory(limit=50000, window_length=self.window_length)
        self.policy = BoltzmannQPolicy()
        self.args = {
            'mode': 'train',
            'env_name': self.env_id,
            'weights': None
        }

        # create the agent
        self.agent = DQNAgent(model=self.model,
                              nb_actions=self.action_space.n,
                              policy=self.policy,
                              memory=self.memory,
                              processor=None,
                              nb_steps_warmup=5000,
                              enable_dueling_network=True,
                              dueling_type='avg',
                              gamma=0.99,
                              target_model_update=1e-2,
                              delta_clip=1.0)

        self.agent.compile(Adam(lr=1e-3), metrics=['mae'])

    def create_model(self):
        features_shape = (self.window_length, self.observation_space.shape[1])
        model = Sequential()
        model.add(CuDNNLSTM(256, input_shape=features_shape))
        model.add(Activation('relu'))

        # model.add(Dropout(0.2))
        #
        # model.add(Dense(128))
        # model.add(Activation('relu'))
        #
        # model.add(Dropout(0.2))
        #
        # model.add(Dense(64))
        # model.add(Activation('relu'))
        #
        # model.add(Dropout(0.2))
        #
        # model.add(Dense(32))
        # model.add(Activation('relu'))

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

            self.agent.fit(self, callbacks=callbacks, nb_steps=100000, log_interval=10000)

            # After training is done, we save the final weights one more time.
            self.agent.save_weights(weights_filename, overwrite=True)

            # Finally, evaluate our algorithm for 10 episodes.
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

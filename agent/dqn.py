from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.optimizers import RMSprop
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
import os
from gym_trading.envs.trading_gym import TradingGym


class Agent(TradingGym):
    name = 'DQN'

    def __init__(self, step_size=1, window_size=5, train=True, max_position=1, weights=True,
                 fitting_file='ETH-USD_2018-12-31.xz', testing_file='ETH-USD_2018-01-01.xz',
                 seed=123,
                 frame_stack=False  # Default to False when using with keras-rl since `rl.memory` stacks frames
                 ):
        super(Agent, self).__init__(training=train,
                                    fitting_file=fitting_file,
                                    testing_file=testing_file,
                                    step_size=step_size,
                                    max_position=max_position,
                                    window_size=window_size,
                                    seed=seed,
                                    frame_stack=False)
        self.memory_frame_stack = 4 if frame_stack else 1  # Number of frames to stack e.g., 4
        self.model = self.create_model()
        self.memory = SequentialMemory(limit=10000, window_length=self.memory_frame_stack)
        self.train = train
        self.training_steps = self.data.shape[0]  # training_steps
        self.weights = weights
        self.cwd = os.path.dirname(os.path.realpath(__file__))

        # create the agent
        self.agent = DQNAgent(model=self.model,
                              nb_actions=self.action_space.n,
                              memory=self.memory,
                              processor=None,
                              nb_steps_warmup=500,
                              enable_dueling_network=True,  # enables double-dueling q-network
                              dueling_type='avg',
                              enable_double_dqn=True,
                              gamma=0.999,
                              target_model_update=1000,
                              delta_clip=1.0)

        self.agent.compile(RMSprop(lr=0.00048), metrics=['mae'])

    def __str__(self):
        # msg = '\n'
        # return msg.join(['{}={}'.format(k, v) for k, v in self.__dict__.items()])
        return 'Agent {} {} | seed {}'.format(Agent.name, self.sym, self.seed())

    def create_model(self):

        features_shape = (self.memory_frame_stack,
                          self.observation_space.shape[0],
                          self.observation_space.shape[1])

        model = Sequential()
        print('agent feature shape: {}'.format(features_shape))
        model.add(Conv2D(input_shape=features_shape,
                         filters=32,
                         kernel_size=8,
                         padding='same',
                         activation='relu',
                         strides=4,
                         data_format='channels_first'))

        model.add(Conv2D(filters=64,
                         kernel_size=4,
                         padding='same',
                         activation='relu',
                         strides=2,
                         data_format='channels_first'))

        model.add(Conv2D(filters=64,
                         kernel_size=3,
                         padding='same',
                         activation='relu',
                         strides=1,
                         data_format='channels_first'))

        model.add(Flatten())

        model.add(Dense(256))
        model.add(Activation('linear'))

        model.add(Dense(self.action_space.n))
        model.add(Activation('softmax'))

        print(model.summary())
        return model

    def start(self):
        weights_filename = '{}/dqn_weights/dqn_{}_weights.h5f'.format(self.cwd, TradingGym.id)
        if self.train:
            if self.weights:
                self.agent.load_weights(weights_filename)
                print('...loading weights for {}'.format(TradingGym.id))

            checkpoint_weights_filename = 'dqn_' + TradingGym.id + '_weights_{step}.h5f'
            checkpoint_weights_filename = '{}/dqn_weights/'.format(self.cwd) + checkpoint_weights_filename
            log_filename = '{}/dqn_weights/dqn_{}_log.json'.format(self.cwd, TradingGym.id)
            print('FileLogger: {}'.format(log_filename))

            callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
            callbacks += [FileLogger(log_filename, interval=100)]

            self.agent.fit(self, callbacks=callbacks, nb_steps=self.training_steps, log_interval=10000, verbose=0)
            self.agent.save_weights(weights_filename, overwrite=True)
            # self.agent.test(self, nb_episodes=2, visualize=False)
            # self.render()
        else:
            self.agent.load_weights(weights_filename)
            self.agent.test(self, nb_episodes=2, visualize=False)
            # self.render()

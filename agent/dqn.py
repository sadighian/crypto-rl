from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.optimizers import RMSprop
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
import os
import gym
import gym_trading


class Agent(object):
    name = 'DQN'

    def __init__(self, step_size=1, window_size=20, train=True, max_position=5,
                 weights=True, fitting_file='ETH-USD_2018-12-31.xz',
                 testing_file='ETH-USD_2018-01-01.xz',
                 format_3d=False,  # add 3rd dimension for CNNs
                 env='market-maker-v0',
                 seed=1,
                 action_repeats=4,
                 number_of_training_steps=1e5,
                 visualize=False):
        self.env_name = env
        self.env = gym.make(self.env_name,
                            training=train,
                            fitting_file=fitting_file,
                            testing_file=testing_file,
                            step_size=step_size,
                            max_position=max_position,
                            window_size=window_size,
                            seed=seed,
                            action_repeats=action_repeats,
                            format_3d=format_3d)
        # Number of frames to stack e.g., 1; Keras-RL uses its own stacker
        self.memory_frame_stack = 1
        self.model = self.create_model()
        self.memory = SequentialMemory(limit=10000,
                                       window_length=self.memory_frame_stack)
        self.train = train
        self.number_of_training_steps = number_of_training_steps
        self.weights = weights
        self.cwd = os.path.dirname(os.path.realpath(__file__))
        self.visualize = visualize

        # create the agent
        self.agent = DQNAgent(model=self.model,
                              nb_actions=self.env.action_space.n,
                              memory=self.memory,
                              processor=None,
                              nb_steps_warmup=500,
                              enable_dueling_network=True,
                              dueling_type='avg',
                              enable_double_dqn=True,
                              gamma=0.999,
                              target_model_update=1000,
                              delta_clip=1.0)

        self.agent.compile(RMSprop(lr=0.00048), metrics=['mae'])

    def __str__(self):
        # msg = '\n'
        # return msg.join(['{}={}'.format(k, v) for k, v in self.__dict__.items()])
        return 'Agent = {} | env = {} | number_of_training_steps = {}'.format(
            Agent.name, self.env_name, self.number_of_training_steps)

    def create_model(self):

        features_shape = (self.memory_frame_stack, *self.env.observation_space.shape)
        model = Sequential()
        conv = Conv2D

        model.add(conv(input_shape=features_shape, filters=16, kernel_size=8,
                       padding='same', activation='relu', strides=4,
                       data_format='channels_first'))
        model.add(conv(filters=32, kernel_size=4, padding='same', activation='relu',
                       strides=2, data_format='channels_first'))
        model.add(conv(filters=32, kernel_size=2, padding='same', activation='relu',
                       strides=1, data_format='channels_first'))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('linear'))
        model.add(Dense(self.env.action_space.n))
        model.add(Activation('softmax'))

        print(model.summary())
        return model

    def start(self):
        weights_filename = '{}/dqn_weights/dqn_{}_weights.h5f'.format(self.cwd,
                                                                      self.env_name)
        if self.weights:
            self.agent.load_weights(weights_filename)
            print('...loading weights for {}'.format(self.env_name))

        if self.train:
            checkpoint_weights_filename = 'dqn_' + self.env_name + \
                                          '_weights_{step}.h5f'
            checkpoint_weights_filename = '{}/dqn_weights/'.format(self.cwd) + \
                                          checkpoint_weights_filename
            log_filename = '{}/dqn_weights/dqn_{}_log.json'.format(self.cwd,
                                                                   self.env_name)
            print('FileLogger: {}'.format(log_filename))

            callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename,
                                                 interval=250000)]
            callbacks += [FileLogger(log_filename, interval=100)]

            print('Starting training...')
            self.agent.fit(self.env,
                           callbacks=callbacks,
                           nb_steps=self.number_of_training_steps,
                           log_interval=10000,
                           verbose=0,
                           visualize=self.visualize)
            print('Saving AGENT weights...')
            self.agent.save_weights(weights_filename, overwrite=True)
        else:
            print('Starting TEST...')
            self.agent.test(self.env, nb_episodes=2, visualize=self.visualize)

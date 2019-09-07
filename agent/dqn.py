from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
import os
import gym
import gym_trading


class Agent(object):
    name = 'DQN'

    def __init__(self, step_size=1, window_size=20, max_position=5,
                 fitting_file='ETH-USD_2018-12-31.xz',
                 testing_file='ETH-USD_2018-01-01.xz',
                 env='market-maker-v0',
                 seed=1,
                 action_repeats=4,
                 number_of_training_steps=1e5,
                 gamma=0.999,
                 format_3d=False,  # add 3rd dimension for CNNs
                 train=True,
                 weights=True,
                 z_score=True,
                 visualize=False,
                 dueling_network=True,
                 double_dqn=True):
        """
        Agent constructor
        :param step_size: int, number of steps to take in env for a given simulation step
        :param window_size: int, number of lags to include in observation
        :param max_position: int, maximum number of positions able to be held in inventory
        :param fitting_file: str, file used for z-score fitting
        :param testing_file: str,file used for dqn experiment
        :param env: environment name
        :param seed: int, random seed number
        :param action_repeats: int, number of steps to take in environment between actions
        :param number_of_training_steps: int, number of steps to train agent for
        :param gamma: float, value between 0 and 1 used to discount future DQN returns
        :param format_3d: boolean, format observation as matrix or tensor
        :param train: boolean, train or test agent
        :param weights: boolean, import existing weights
        :param z_score: boolean, standardize observation space
        :param visualize: boolean, visiualize environment
        :param dueling_network: boolean, use dueling network architecture
        :param double_dqn: boolean, use double DQN for Q-value approximation
        """
        self.env_name = env
        self.env = gym.make(self.env_name,
                            fitting_file=fitting_file,
                            testing_file=testing_file,
                            step_size=step_size,
                            max_position=max_position,
                            window_size=window_size,
                            seed=seed,
                            action_repeats=action_repeats,
                            training=train,
                            z_score=z_score,
                            format_3d=format_3d)
        # Number of frames to stack e.g., 1.
        # NOTE: 'Keras-RL' uses its own frame-stacker
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
                              enable_dueling_network=dueling_network,
                              dueling_type='avg',
                              enable_double_dqn=double_dqn,
                              gamma=gamma,
                              target_model_update=1000,
                              delta_clip=1.0)
        self.agent.compile(Adam(lr=float("3e-4")), metrics=['mae'])

    def __str__(self):
        # msg = '\n'
        # return msg.join(['{}={}'.format(k, v) for k, v in self.__dict__.items()])
        return 'Agent = {} | env = {} | number_of_training_steps = {}'.format(
            Agent.name, self.env_name, self.number_of_training_steps)

    def create_model(self):
        """
        Create a Convolutional neural network with dense layer at the end
        :return: keras model
        """
        features_shape = (self.memory_frame_stack, *self.env.observation_space.shape)
        model = Sequential()
        conv = Conv2D

        model.add(conv(input_shape=features_shape,
                       filters=16, kernel_size=[10, 1], padding='same', activation='relu',
                       strides=[5, 1], data_format='channels_first'))
        model.add(conv(filters=16, kernel_size=[6, 1], padding='same', activation='relu',
                       strides=[3, 1], data_format='channels_first'))
        model.add(conv(filters=16, kernel_size=[4, 1], padding='same', activation='relu',
                       strides=[2, 1], data_format='channels_first'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('linear'))
        model.add(Dense(self.env.action_space.n))
        model.add(Activation('softmax'))

        print(model.summary())
        return model

    def start(self):
        """
        Entry point for agent training and testing
        :return: (void)
        """
        weights_filename = '{}/dqn_weights/dqn_{}_weights.h5f'.format(
            self.cwd, self.env_name)

        if self.weights:
            self.agent.load_weights(weights_filename)
            print('...loading weights for {}'.format(self.env_name))

        if self.train:
            checkpoint_weights_filename = 'dqn_{}'.format(self.env_name) + \
                                          '_weights_{step}.h5f'
            checkpoint_weights_filename = '{}/dqn_weights/'.format(self.cwd) + \
                                          checkpoint_weights_filename
            log_filename = '{}/dqn_weights/dqn_{}_log.json'.format(
                self.cwd, self.env_name)
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

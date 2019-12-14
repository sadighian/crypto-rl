from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from configurations import LOGGER
import os
import gym
import gym_trading


class Agent(object):
    name = 'DQN'

    def __init__(self, number_of_training_steps=1e5, gamma=0.999, load_weights=False,
                 visualize=False, dueling_network=True, double_dqn=True, nn_type='mlp',
                 **kwargs):
        """
        Agent constructor
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
        :param load_weights: boolean, import existing weights
        :param z_score: boolean, standardize observation space
        :param visualize: boolean, visualize environment
        :param dueling_network: boolean, use dueling network architecture
        :param double_dqn: boolean, use double DQN for Q-value approximation
        """
        # Agent arguments
        # self.env_name = id
        self.neural_network_type = nn_type
        self.load_weights = load_weights
        self.number_of_training_steps = number_of_training_steps
        self.visualize = visualize

        # Create environment
        self.env = gym.make(**kwargs)
        self.env_name = self.env.env.id

        # Create agent
        # NOTE: 'Keras-RL' uses its own frame-stacker
        self.memory_frame_stack = 1  # Number of frames to stack e.g., 1.
        self.model = self.create_model(name=self.neural_network_type)
        self.memory = SequentialMemory(limit=10000,
                                       window_length=self.memory_frame_stack)
        self.train = self.env.env.training
        self.cwd = os.path.dirname(os.path.realpath(__file__))

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

    def create_model(self, name: str = 'cnn') -> Sequential:
        """
        Helper function get create and get the default MLP or CNN model.

        :param name: Neural network type ['mlp' or 'cnn']
        :return: neural network
        """
        LOGGER.info("creating model for {}".format(name))
        if name == 'cnn':
            return self._create_cnn_model()
        elif name == 'mlp':
            return self._create_mlp_model()

    def _create_cnn_model(self) -> Sequential:
        """
        Create a Convolutional neural network with dense layer at the end.

        :return: keras model
        """
        features_shape = (self.memory_frame_stack, *self.env.observation_space.shape)
        model = Sequential()
        conv = Conv2D
        model.add(conv(input_shape=features_shape,
                       filters=5, kernel_size=[10, 1], padding='same', activation='relu',
                       strides=[5, 1], data_format='channels_first'))
        model.add(conv(filters=5, kernel_size=[5, 1], padding='same', activation='relu',
                       strides=[2, 1], data_format='channels_first'))
        model.add(conv(filters=5, kernel_size=[4, 1], padding='same', activation='relu',
                       strides=[2, 1], data_format='channels_first'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='softmax'))
        LOGGER.info(model.summary())
        return model

    def _create_mlp_model(self) -> Sequential:
        """
        Create a DENSE neural network with dense layer at the end

        :return: keras model
        """
        features_shape = (self.memory_frame_stack, *self.env.observation_space.shape)
        model = Sequential()
        model.add(Dense(units=256, input_shape=features_shape, activation='relu'))
        model.add(Dense(units=256, activation='relu'))
        model.add(Flatten())
        model.add(Dense(self.env.action_space.n, activation='softmax'))
        LOGGER.info(model.summary())
        return model

    def start(self) -> None:
        """
        Entry point for agent training and testing

        :return: (void)
        """
        output_directory = os.path.join(self.cwd, 'dqn_weights')
        if not os.path.exists(output_directory):
            LOGGER.info('{} does not exist. Creating Directory.'.format(output_directory))
            os.mkdir(output_directory)

        weight_name = 'dqn_{}_{}_weights.h5f'.format(
            self.env_name, self.neural_network_type)
        weights_filename = os.path.join(output_directory, weight_name)
        LOGGER.info("weights_filename: {}".format(weights_filename))

        if self.load_weights:
            LOGGER.info('...loading weights for {} from\n{}'.format(
                self.env_name, weights_filename))
            self.agent.load_weights(weights_filename)

        if self.train:
            step_chkpt = '{step}.h5f'
            step_chkpt = 'dqn_{}_weights_{}'.format(self.env_name, step_chkpt)
            checkpoint_weights_filename = os.path.join(self.cwd,
                                                       'dqn_weights',
                                                       step_chkpt)
            LOGGER.info("checkpoint_weights_filename: {}".format(
                checkpoint_weights_filename))
            log_filename = os.path.join(self.cwd, 'dqn_weights',
                                        'dqn_{}_log.json'.format(self.env_name))
            LOGGER.info('log_filename: {}'.format(log_filename))

            callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename,
                                                 interval=250000)]
            callbacks += [FileLogger(log_filename, interval=100)]

            LOGGER.info('Starting training...')
            self.agent.fit(self.env,
                           callbacks=callbacks,
                           nb_steps=self.number_of_training_steps,
                           log_interval=10000,
                           verbose=0,
                           visualize=self.visualize)
            LOGGER.info("training over.")
            LOGGER.info('Saving AGENT weights...')
            self.agent.save_weights(weights_filename, overwrite=True)
            LOGGER.info("AGENT weights saved.")
        else:
            LOGGER.info('Starting TEST...')
            self.agent.test(self.env, nb_episodes=2, visualize=self.visualize)

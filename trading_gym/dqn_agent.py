import argparse

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Input, Conv1D, CuDNNLSTM, Dropout, Permute
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from trading_gym import TradingGym


def create_model(input_shape=(1, 128), action_space=3):
    # Create the network model
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(action_space))
    model.add(Activation('softmax'))
    print(model.summary())
    return model


def create_agent(input_shape=(1, 128), action_space=3, window_length=4):
    memory = SequentialMemory(limit=1000000, window_length=window_length)
    # processor = Processor()

    policy = BoltzmannQPolicy()

    model = create_model(input_shape=input_shape, action_space=action_space)

    dqn = DQNAgent(model=model,
                   nb_actions=action_space,
                   policy=policy,
                   memory=memory,
                   processor=None,
                   nb_steps_warmup=50,
                   enable_dueling_network=True,
                   dueling_type='avg',
                   gamma=0.99,
                   target_model_update=1e-2,
                   delta_clip=1.0)

    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    return dqn


def train():
    env = TradingGym()

    INPUT_SHAPE = env.observation_space.shape
    print('INPUT_SHAPE: {}'.format(INPUT_SHAPE))

    WINDOW_LENGTH = 4
    print('WINDOW_LENGTH: {}'.format(WINDOW_LENGTH))

    args = {
        'mode': 'train',
        'env_name': env.env_id,
        'weights': None
    }

    input_shape = (WINDOW_LENGTH, INPUT_SHAPE[1])
    print('input_shape: {}'.format(input_shape))

    dqn = create_agent(input_shape=input_shape,
                       action_space=env.action_space.n,
                       window_length=WINDOW_LENGTH)

    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    weights_filename = 'dqn_{}_weights.h5f'.format(args['env_name'])
    checkpoint_weights_filename = 'dqn_' + args['env_name'] + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args['env_name'])

    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]

    dqn.fit(env, callbacks=callbacks, nb_steps=150000, log_interval=10000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=3, visualize=False)


if __name__ == '__main__':
    print('training...')
    train()
    print('...done training.')

import gym
import numpy as np
from tensorforce import TensorForceError
from tensorforce.environments import Environment

from trading_gym import TradingGym


class TradingEnv(Environment):

    def __init__(self, query, lags):
        self.gym = TradingGym(query, lags)
        # self._states = dict(type='float', shape=self.gym.observation_space.shape)
        # self._actions = self.actions_ = dict(type='int', shape=(), num_actions=len(self.gym.actions))
        self._states = TradingEnv.state_from_space(space=self.gym.observation_space)
        self._actions = TradingEnv.action_from_space(space=self.gym.action_space)

    def __str__(self):
        return 'Simulation Environment: {}'.format(self.gym.env_id)

    @property
    def states(self):
        return self._states

    @property
    def actions(self):
        return self._actions

    def close(self):
        self.gym.close()
        self.gym = None

    def reset(self):
        return self.gym.reset()

    def execute(self, action):
        observation, reward, done, info = self.gym.step(action)
        return observation, reward, done

    @staticmethod
    def state_from_space(space):
        if isinstance(space, gym.spaces.Discrete):
            return dict(shape=(), type='int')
        elif isinstance(space, gym.spaces.MultiBinary):
            return dict(shape=space.n, type='int')
        elif isinstance(space, gym.spaces.MultiDiscrete):
            return dict(shape=space.num_discrete_space, type='int')
        elif isinstance(space, gym.spaces.Box):
            return dict(shape=tuple(space.shape), type='float')
        elif isinstance(space, gym.spaces.Tuple):
            states = dict()
            n = 0
            for n, space in enumerate(space.spaces):
                state = TradingEnv.state_from_space(space=space)
                if 'type' in state:
                    states['gymtpl{}'.format(n)] = state
                else:
                    for name, state in state.items():
                        states['gymtpl{}-{}'.format(n, name)] = state
            return states
        elif isinstance(space, gym.spaces.Dict):
            states = dict()
            for space_name, space in space.spaces.items():
                state = TradingEnv.state_from_space(space=space)
                if 'type' in state:
                    states[space_name] = state
                else:
                    for name, state in state.items():
                        states['{}-{}'.format(space_name, name)] = state
            return states
        else:
            raise TensorForceError('Unknown Gym space.')

    @staticmethod
    def flatten_state(state):
        if isinstance(state, tuple):
            states = dict()
            for n, state in enumerate(state):
                state = TradingEnv.flatten_state(state=state)
                if isinstance(state, dict):
                    for name, state in state.items():
                        states['gymtpl{}-{}'.format(n, name)] = state
                else:
                    states['gymtpl{}'.format(n)] = state
            return states
        elif isinstance(state, dict):
            states = dict()
            for state_name, state in state.items():
                state = TradingEnv.flatten_state(state=state)
                if isinstance(state, dict):
                    for name, state in state.items():
                        states['{}-{}'.format(state_name, name)] = state
                else:
                    states['{}'.format(state_name)] = state
            return states
        else:
            return state

    @staticmethod
    def action_from_space(space):
        if isinstance(space, gym.spaces.Discrete):
            return dict(type='int', num_actions=space.n)
        elif isinstance(space, gym.spaces.MultiBinary):
            return dict(type='bool', shape=space.n)
        elif isinstance(space, gym.spaces.MultiDiscrete):
            num_discrete_space = len(space.nvec)
            if (space.nvec == space.nvec[0]).all():
                return dict(type='int', num_actions=space.nvec[0], shape=num_discrete_space)
            else:
                actions = dict()
                for n in range(num_discrete_space):
                    actions['gymmdc{}'.format(n)] = dict(type='int', num_actions=space.nvec[n])
                return actions
        elif isinstance(space, gym.spaces.Box):
            if (space.low == space.low[0]).all() and (space.high == space.high[0]).all():
                return dict(type='float', shape=space.low.shape,
                            min_value=np.float32(space.low[0]),
                            max_value=np.float32(space.high[0]))
            else:
                actions = dict()
                low = space.low.flatten()
                high = space.high.flatten()
                for n in range(low.shape[0]):
                    actions['gymbox{}'.format(n)] = dict(type='float', min_value=low[n], max_value=high[n])
                return actions
        elif isinstance(space, gym.spaces.Tuple):
            actions = dict()
            n = 0
            for n, space in enumerate(space.spaces):
                action = TradingEnv.action_from_space(space=space)
                if 'type' in action:
                    actions['gymtpl{}'.format(n)] = action
                else:
                    for name, action in action.items():
                        actions['gymtpl{}-{}'.format(n, name)] = action
            return actions
        elif isinstance(space, gym.spaces.Dict):
            actions = dict()
            for space_name, space in space.spaces.items():
                action = TradingEnv.action_from_space(space=space)
                if 'type' in action:
                    actions[space_name] = action
                else:
                    for name, action in action.items():
                        actions['{}-{}'.format(space_name, name)] = action
            return actions

        else:
            raise TensorForceError('Unknown Gym space.')

    @staticmethod
    def unflatten_action(action):
        if not isinstance(action, dict):
            return action
        elif all(name.startswith('gymmdc') for name in action) or \
                all(name.startswith('gymbox') for name in action) or \
                all(name.startswith('gymtpl') for name in action):
            space_type = next(iter(action))[:6]
            actions = list()
            n = 0
            while True:
                if any(name.startswith(space_type + str(n) + '-') for name in action):
                    inner_action = {
                        name[name.index('-') + 1:] for name, inner_action in action.items()
                        if name.startswith(space_type + str(n))
                    }
                    actions.append(TradingEnv.unflatten_action(action=inner_action))
                elif any(name == space_type + str(n) for name in action):
                    actions.append(action[space_type + str(n)])
                else:
                    break
                n += 1
            return tuple(actions)
        else:
            actions = dict()
            for name, action in action.items():
                if '-' in name:
                    name, inner_name = name.split('-', 1)
                    if name not in actions:
                        actions[name] = dict()
                    actions[name][inner_name] = action
                else:
                    actions[name] = action
            for name, action in actions.items():
                if isinstance(action, dict):
                    actions[name] = TradingEnv.unflatten_action(action=action)
            return actions


def make(query, lags):
    return TradingEnv(query, lags)

from gym.envs.registration import register
from gym_trading.envs.trading_gym import TradingGym

register(
    id=TradingGym.id,
    entry_point='gym_trading.envs:TradingGym',
    max_episode_steps=1000000,
    nondeterministic=False
)

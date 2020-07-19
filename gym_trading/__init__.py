from gym.envs.registration import register

from gym_trading.envs.market_maker import MarketMaker
from gym_trading.envs.trend_following import TrendFollowing

register(
    id=TrendFollowing.id,
    entry_point='gym_trading.envs:TrendFollowing',
    max_episode_steps=1000000,
    nondeterministic=False
)

register(
    id=MarketMaker.id,
    entry_point='gym_trading.envs:MarketMaker',
    max_episode_steps=1000000,
    nondeterministic=False
)
